import importlib.util
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf


TRAINING_DIR = Path(__file__).resolve().parents[1]
LAUNCHER = TRAINING_DIR / "scripts" / "train_prco_7b.sh"


class ReleaseLauncherTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory(prefix="prco release ")
        self.addCleanup(self.workspace.cleanup)
        self.workspace_path = Path(self.workspace.name)
        self.environment = os.environ.copy()
        for name in (
            "DATA_ROOT",
            "TRAIN_JSONL",
            "VAL_JSONL",
            "TRAIN_IMAGE_DIR",
            "VAL_IMAGE_DIR",
            "OUTPUT_DIR",
            "MODEL_PATH",
            "PYTHON_BIN",
            "VLLM_JUDGE_MODEL",
        ):
            self.environment.pop(name, None)
        self.environment.update(VLLM_JUDGE_IP="127.0.0.1", VLLM_JUDGE_PORTS="9200")

    def run_launcher(self, *arguments):
        return subprocess.run(
            ["bash", str(LAUNCHER), *arguments],
            cwd=self.workspace_path,
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_dry_run_resolves_relative_paths_without_side_effects(self):
        self.environment.pop("VLLM_JUDGE_IP")
        self.environment.pop("VLLM_JUDGE_PORTS")
        self.environment.update(DATA_ROOT="data with spaces", OUTPUT_DIR="run with spaces")
        result = self.run_launcher("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        command = shlex.split(result.stdout.split("Command: ", 1)[1])
        self.assertIn(f"data.image_dir={self.workspace_path}/data with spaces/train_images", command)
        self.assertIn(f"trainer.save_checkpoint_path={self.workspace_path}/run with spaces/checkpoints", command)
        self.assertEqual(list(self.workspace_path.iterdir()), [])

    def test_default_paths_resolve_from_invocation_directory(self):
        result = self.run_launcher("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        command = shlex.split(result.stdout.split("Command: ", 1)[1])
        overrides = OmegaConf.from_dotlist(command[3:])
        self.assertEqual(Path(overrides.data.train_files), self.workspace_path / "data/train.jsonl")
        self.assertEqual(Path(overrides.data.val_files), self.workspace_path / "data/val.jsonl")
        self.assertEqual(Path(overrides.data.image_dir), self.workspace_path / "data/train_images")
        self.assertEqual(Path(overrides.data.val_image_dir), self.workspace_path / "data/val_images")
        self.assertEqual(
            Path(overrides.trainer.save_checkpoint_path), self.workspace_path / "outputs/prco_7b/checkpoints"
        )

    def test_training_requires_explicit_judge_endpoint(self):
        for name, value in (("VLLM_JUDGE_IP", "127.0.0.1"), ("VLLM_JUDGE_PORTS", "9200")):
            with self.subTest(name=name):
                self.environment.pop(name)
                result = self.run_launcher()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(name, result.stderr)
                self.environment[name] = value

    def test_missing_dataset_fails_before_loading_training(self):
        self.environment["DATA_ROOT"] = str(self.workspace_path / "missing")
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Dataset file not found", result.stderr)

    def test_unknown_option_is_rejected(self):
        result = self.run_launcher("--unknown")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Usage:", result.stderr)

    def test_launch_preserves_recipe_and_clears_unused_switches(self):
        for directory in ("data/train_images", "data/val_images", "local model"):
            (self.workspace_path / directory).mkdir(parents=True)
        for dataset in ("data/train.jsonl", "data/val.jsonl"):
            (self.workspace_path / dataset).write_text("{}\n", encoding="utf-8")
        python_stub = self.workspace_path / "python_stub"
        python_stub.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "if sys.argv[1] == '-c':\n"
            "    print(8)\n"
            "else:\n"
            "    names = ['DRGRPO', 'caption_rollout_n', 'CAPTION_ROLLOUT_N', 'use_raw', 'vllm_judge', "
            "'vllm_judge_use_images', 'vllm_judge_ip', 'vllm_judge_ports', 'vllm_judge_model', "
            "'discard', 'DISCARD', 'vllm_judge_discard', "
            "'VLLM_JUDGE_DISCARD', 'skip_caption_rollout_update', 'SKIP_CAPTION_ROLLOUT_UPDATE', "
            "'use_var_max', 'USE_VAR_MAX']\n"
            "    payload = {'args': sys.argv[1:], 'cwd': os.getcwd(), "
            "'environment': {name: os.getenv(name) for name in names}}\n"
            "    Path(os.environ['CAPTURE_PATH']).write_text(json.dumps(payload))\n",
            encoding="utf-8",
        )
        python_stub.chmod(0o755)
        capture_path = self.workspace_path / "captured.json"
        self.environment.update(
            DATA_ROOT="data",
            OUTPUT_DIR="run",
            MODEL_PATH="local model",
            PYTHON_BIN=str(python_stub),
            CAPTURE_PATH=str(capture_path),
            VLLM_JUDGE_IP="192.0.2.10",
            VLLM_JUDGE_PORTS="8100,8101",
            DRGRPO="0",
            caption_rollout_n="2",
            CAPTION_ROLLOUT_N="2",
            use_raw="false",
            discard="true",
            DISCARD="true",
            vllm_judge_discard="true",
            VLLM_JUDGE_DISCARD="true",
            skip_caption_rollout_update="true",
            SKIP_CAPTION_ROLLOUT_UPDATE="true",
            use_var_max="true",
            USE_VAR_MAX="true",
        )
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        captured = json.loads(capture_path.read_text())
        self.assertEqual(captured["cwd"], str(TRAINING_DIR))
        overrides = OmegaConf.from_dotlist(captured["args"][2:])
        overrides.pop("config")
        from verl.trainer.config import PPOConfig

        config = OmegaConf.to_object(
            OmegaConf.merge(
                OmegaConf.structured(PPOConfig()),
                OmegaConf.load(TRAINING_DIR / "configs/prco_7b.yaml"),
                overrides,
            )
        )
        self.assertEqual(config.trainer.max_steps, 200)
        self.assertEqual(config.trainer.experiment_name, "prco_7b")
        self.assertEqual(config.worker.actor.model.model_path, str(self.workspace_path / "local model"))
        self.assertEqual(config.worker.rollout.n, 8)
        self.assertEqual(config.data.solve_use_images_switch_step, 40)
        self.assertTrue(config.algorithm.disable_kl)
        self.assertTrue(config.algorithm.online_filtering)
        self.assertFalse(config.worker.actor.use_role_clip)
        self.assertEqual(config.worker.reward.reward_function_kwargs, {})
        self.assertEqual(config.worker.actor.clip_ratio_high, 0.28)
        self.assertEqual(config.worker.actor.global_batch_size, 96)
        environment = captured["environment"]
        for name, value in {
            "DRGRPO": "1",
            "caption_rollout_n": "4",
            "use_raw": "true",
            "vllm_judge": "true",
            "vllm_judge_use_images": "false",
            "vllm_judge_ip": "192.0.2.10",
            "vllm_judge_ports": "8100,8101",
            "vllm_judge_model": "Qwen/Qwen3-VL-8B-Instruct",
        }.items():
            self.assertEqual(environment.pop(name), value)
        self.assertTrue(all(value is None for value in environment.values()))


class JudgeLogTests(unittest.TestCase):
    def test_judge_logs_follow_output_directory_without_network_calls(self):
        spec = importlib.util.spec_from_file_location(
            "prco_release_judge", TRAINING_DIR / "verl/workers/reward/vllm_judge.py"
        )
        judge_module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {spec.name: judge_module}):
            spec.loader.exec_module(judge_module)
            with tempfile.TemporaryDirectory() as log_dir:
                with patch.dict(os.environ, {"PRCO_LOG_DIR": log_dir, "EXPERIMENT_NAME": "prco_7b"}):
                    config = judge_module.VLLMJudgeConfig(
                        ip="127.0.0.1",
                        model="test-judge",
                        template_path=str(TRAINING_DIR / "prompts/judge.jinja"),
                    )
                    judge = judge_module.VLLMCaptionJudge(config)
                    self.assertTrue(Path(judge._log_path).is_relative_to(log_dir))
                    self.assertTrue(Path(judge._score_zero_log_path).is_relative_to(log_dir))
                    judge._maybe_log_score_zero(prompt_text="test", response_text='{"label": 0}', label=0)
                    self.assertTrue(Path(judge._score_zero_log_path).is_file())


if __name__ == "__main__":
    unittest.main()
