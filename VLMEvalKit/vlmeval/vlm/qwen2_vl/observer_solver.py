from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..base import BaseModel
from .model import Qwen2VLChat


def _log_prompt_record(record: Dict[str, Any]) -> None:
    """Append a JSON line to LOG_PROMPT_FILE if set (best-effort)."""

    log_path = os.getenv("LOG_PROMPT_FILE")
    if not log_path:
        return

    try:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as f:
            try:
                import fcntl  # type: ignore

                fcntl.flock(f, fcntl.LOCK_EX)
            except Exception:
                pass
            f.write(line + "\n")
            try:
                fcntl.flock(f, fcntl.LOCK_UN)
            except Exception:
                pass
    except Exception as e:  # noqa: BLE001
        print(f"[LOG_PROMPT_FILE] write failed: {e}", file=sys.stderr)


def _get_visible_cuda_devices() -> List[str]:
    """Return visible CUDA devices in CUDA_VISIBLE_DEVICES order.

    - If CUDA_VISIBLE_DEVICES is set, returns it (split by comma).
    - Otherwise falls back to `nvidia-smi --list-gpus`.
    """
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cvd:
        return [d.strip() for d in cvd.split(',') if d.strip()]
    try:
        out = subprocess.check_output(['nvidia-smi', '--list-gpus'], stderr=subprocess.DEVNULL)
        lines = [ln for ln in out.decode('utf-8', errors='ignore').splitlines() if ln.strip()]
        return [str(i) for i in range(len(lines))]
    except Exception:
        return []


_DEFAULT_OBSERVER_TEMPLATE = """You are given an image and a relevant question. Based on the query, please describe the image in detail.
Do not try to answer the question.

Question: {{ content | trim }}

Please only describe the image. DO NOT try to answer the question!"""

_DEFAULT_SOLVER_TEMPLATE = """You are a helpful assistant.

### The detailed caption of the provided image: {{ content.caption | trim }}
### Question: {{ content.question | trim }}

Please think step by step. The final answer MUST BE put in \\boxed{}."""

_LOG_PROMPT_COUNTER = 0


class Qwen2VLObserverSolver(BaseModel):
    """Two-stage (Observer -> Solver) wrapper for Qwen2.5-VL models.

    This wrapper runs:
        1) Observer: generate a detailed caption conditioned on the image + question.
        2) Solver: answer the question conditioned on the same image + observer caption.

    It is designed to match the user's training rollout input constraints:
        - single-turn
        - images are placed *before* the text (no interleaving)
        - templates are Jinja2 templates (defaults provided, can be overridden via file paths or strings)
    """

    INSTALL_REQ = False
    INTERLEAVE = False
    allowed_types = ['text', 'image']

    def __init__(
        self,
        observer_model_path: Optional[str] = None,
        solver_model_path: Optional[str] = None,
        observer: Optional[Dict[str, Any]] = None,
        solver: Optional[Dict[str, Any]] = None,
        observer_template: Optional[str] = None,
        solver_template: Optional[str] = None,
        observer_template_path: Optional[str] = None,
        solver_template_path: Optional[str] = None,
        # When True, will enable Qwen2VLChat.post_process for the solver to extract \\boxed{...}
        solver_post_process: bool = True,
        # Propagate vLLM switch to both sub-models unless overridden in sub-config.
        use_vllm: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # Backward/alias support: allow passing either (observer_model_path, solver_model_path)
        # or nested dict configs via `observer`/`solver`.
        if observer is None:
            observer = {}
        if solver is None:
            solver = {}

        # Default gpu_utils depends on available GPUs:
        # - single GPU: be conservative (0.45) to avoid OOM
        # - two or more GPUs: allow higher util (0.9)
        visible_devices = _get_visible_cuda_devices()
        if len(visible_devices) == 1:
            observer.setdefault('gpu_utils', 0.45)
            solver.setdefault('gpu_utils', 0.45)
        elif len(visible_devices) >= 2:
            observer.setdefault('gpu_utils', 0.9)
            solver.setdefault('gpu_utils', 0.9)

        if observer_model_path is not None:
            observer = dict(observer)
            observer.setdefault('model_path', observer_model_path)
        if solver_model_path is not None:
            solver = dict(solver)
            solver.setdefault('model_path', solver_model_path)

        if 'model_path' not in observer or 'model_path' not in solver:
            raise ValueError(
                'Qwen2VLObserverSolver requires both observer and solver model paths. '
                'Provide `observer_model_path` & `solver_model_path`, or pass dicts `observer`/`solver` with `model_path`.'
            )

        # Templates: allow overriding by file path or string.
        self.observer_template = self._load_template(
            template=observer_template,
            template_path=observer_template_path,
            default=_DEFAULT_OBSERVER_TEMPLATE,
        )
        self.solver_template = self._load_template(
            template=solver_template,
            template_path=solver_template_path,
            default=_DEFAULT_SOLVER_TEMPLATE,
        )

        # Make vLLM visible in nested configs unless user explicitly sets it there.
        observer.setdefault('use_vllm', use_vllm)
        solver.setdefault('use_vllm', use_vllm)

        # Expose a top-level flag so the inference loop can detect vLLM usage.
        self.use_vllm = bool(observer.get('use_vllm', False)) or bool(solver.get('use_vllm', False))

        # Solver post-process default for math-style answers.
        solver.setdefault('post_process', solver_post_process)

        # Some global kwargs might be intended for both sub-models; copy them if not present.
        # Examples: min_pixels/max_pixels/total_pixels, temperature, max_new_tokens, etc.
        for k, v in kwargs.items():
            observer.setdefault(k, v)
            solver.setdefault(k, v)

        # Instantiate underlying models (Qwen2.5-VL chat impl already supports vLLM).
        #
        # If exactly 2 GPUs are visible, pin Observer/Solver vLLM engines onto different GPUs
        # (Observer -> first GPU, Solver -> second GPU). This avoids the default behaviour where
        # each sub-model may try to use all visible GPUs.
        use_vllm_obs = bool(observer.get('use_vllm', False))
        use_vllm_sol = bool(solver.get('use_vllm', False))
        visible_devices = _get_visible_cuda_devices() if (use_vllm_obs and use_vllm_sol) else []
        if len(visible_devices) == 2:
            obs_dev, sol_dev = visible_devices[0], visible_devices[1]
            cvd_bak = os.environ.get('CUDA_VISIBLE_DEVICES')
            print(
                f"[Qwen2VLObserverSolver] Detected 2 GPUs (CUDA_VISIBLE_DEVICES={cvd_bak or 'unset'}). "
                f"Pinning observer vLLM to GPU {obs_dev} and solver vLLM to GPU {sol_dev}.",
                flush=True,
            )
            try:
                os.environ['CUDA_VISIBLE_DEVICES'] = obs_dev
                self.observer_model = Qwen2VLChat(**observer)
                os.environ['CUDA_VISIBLE_DEVICES'] = sol_dev
                self.solver_model = Qwen2VLChat(**solver)
            finally:
                if cvd_bak is None:
                    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
                else:
                    os.environ['CUDA_VISIBLE_DEVICES'] = cvd_bak
        else:
            self.observer_model = Qwen2VLChat(**observer)
            self.solver_model = Qwen2VLChat(**solver)

        # Best-effort cap for batch inference (used by the inference loop).
        if getattr(self, 'use_vllm', False):
            obs_mns = getattr(self.observer_model, 'max_num_seqs', None)
            sol_mns = getattr(self.solver_model, 'max_num_seqs', None)
            if obs_mns is not None and sol_mns is not None:
                self.max_num_seqs = int(min(obs_mns, sol_mns))
            elif obs_mns is not None:
                self.max_num_seqs = int(obs_mns)
            elif sol_mns is not None:
                self.max_num_seqs = int(sol_mns)

    def set_dump_image(self, dump_image_func):
        super().set_dump_image(dump_image_func)
        # Not strictly required (we pass image paths directly), but helps if user calls build_prompt on sub-models.
        if hasattr(self, 'observer_model') and self.observer_model is not None:
            self.observer_model.set_dump_image(dump_image_func)
        if hasattr(self, 'solver_model') and self.solver_model is not None:
            self.solver_model.set_dump_image(dump_image_func)

    @staticmethod
    def _load_template(template: Optional[str], template_path: Optional[str], default: str) -> str:
        if template is not None:
            return template
        if template_path is not None:
            with open(os.path.expanduser(template_path), 'r', encoding='utf-8') as f:
                return f.read()
        return default

    @staticmethod
    def _strip_image_placeholders(text: str) -> str:
        # Handle common placeholders used in datasets / prompts.
        # Examples: "<image>", "<image 1>", "<image1>", etc.
        text = re.sub(r'<\s*image\s*\d*\s*>', '', text, flags=re.IGNORECASE)
        # Some prompts may use "image token" formats like "<image 1>" without closing ">" issues.
        text = text.replace('<image>', '')
        return text

    @staticmethod
    def _normalize_question(
        text: str,
        strip_question_prefix: bool = True,
        collapse_blank_lines: bool = True,
    ) -> str:
        text = text.strip()
        text = Qwen2VLObserverSolver._strip_image_placeholders(text)
        # If the dataset prompt already starts with "Question:", avoid duplicating the template's prefix.
        if strip_question_prefix:
            text = re.sub(r'^\s*Question\s*:\s*', '', text, flags=re.IGNORECASE)
        # Collapse excessive blank lines.
        if collapse_blank_lines:
            text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @staticmethod
    def _extract_question_stem(question: str, dataset: Optional[str]) -> str:
        """Extract the bare question statement from dataset-specific prompt templates.

        This is ONLY applied in the Observer->Solver wrapper to avoid double-wrapping
        dataset-provided instruction templates (Hint/Guide/output-format requirements)
        with our fixed Jinja inference prompts.
        """
        q = (question or "").strip()
        if not dataset:
            return q

        ds_lower = str(dataset).lower()

        def after_last(pattern: str, text: str) -> Optional[str]:
            matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
            if not matches:
                return None
            return text[matches[-1].end():]

        def after_first(pattern: str, text: str) -> Optional[str]:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if not m:
                return None
            return text[m.end():]

        def before_first(pattern: str, text: str) -> str:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if not m:
                return text
            return text[:m.start()]

        def finalize(extracted: Optional[str]) -> str:
            if extracted is None:
                return q
            extracted = extracted.strip()
            return extracted if extracted else q

        # MathVerse family: most questions are like "... Question: <stem>"
        if ds_lower.startswith("mathverse"):
            return finalize(after_last(r"\bQuestion\s*:\s*", q))

        # MathVista_MINI: often "Hint: ...\nQuestion: <stem>"
        if ds_lower.startswith("mathvista"):
            cand = after_last(r"\bQuestion\s*:\s*", q)
            if cand is not None and cand.strip():
                return cand.strip()
            # Fallback: drop a leading "Hint: ..." line if present.
            q2 = re.sub(r"^\s*Hint\s*:\s*.*?(?:\n|$)", "", q, flags=re.IGNORECASE).strip()
            cand2 = after_last(r"\bQuestion\s*:\s*", q2)
            return finalize(cand2 if cand2 is not None else q2)

        # MathVision(_MINI): dataset prompt may wrap as:
        # "... Question: {line['question']}\nAnswer:"
        if ds_lower.startswith("mathvision"):
            cand = after_last(r"\bQuestion\s*:\s*", q)
            if cand is None:
                return q
            cand = before_first(r"\bAnswer\s*:\s*", cand).strip()

            # If the embedded question itself contains "Hint: ...\nQuestion: ...",
            # prefer the innermost Question: segment.
            cand2 = after_last(r"\bQuestion\s*:\s*", cand)
            if cand2 is not None and cand2.strip():
                cand = cand2.strip()

            if re.match(r"^\s*Hint\s*:\s*", cand, flags=re.IGNORECASE):
                # Remove leading "Hint:" segment (usually one line).
                cand_wo_hint = re.sub(r"^\s*Hint\s*:\s*", "", cand, flags=re.IGNORECASE)
                cand3 = after_last(r"\bQuestion\s*:\s*", cand_wo_hint)
                if cand3 is not None and cand3.strip():
                    cand = cand3.strip()
                else:
                    # Otherwise, drop the remainder of the hint line.
                    cand = re.sub(r"^.*?\n", "", cand_wo_hint).strip()

            return finalize(cand)

        # WeMath / WeMath_COT: usually "Hint: ...\nQuestion: <stem>\nOptions: ..."
        if ds_lower.startswith("wemath"):
            # NOTE: WeMath is MCQ. The "Options:" block is essential information for solving.
            # We therefore only strip the leading "Hint:" wrapper, but keep the rest (Question + Options [+ requirement]).
            q2 = re.sub(r"^\s*Hint\s*:\s*.*?(?:\n|$)", "", q, flags=re.IGNORECASE).strip()
            # Let _normalize_question() remove a leading "Question:" prefix if it becomes the first line.
            return q2 if q2 else q

        # DynaMath: prompt starts with "## Question\n <stem>" then adds a guide section.
        if ds_lower.startswith("dynamath"):
            cand = q
            m = re.search(r"##\s*Question\b", cand, flags=re.IGNORECASE)
            if m:
                cand = cand[m.end():]
            cand = cand.lstrip()
            cand = before_first(r"##\s*Answer\s*Instruction\b", cand)
            return finalize(cand)

        return q

    @staticmethod
    def _extract_images_and_text(message: List[Dict[str, Any]]) -> Tuple[List[str], str]:
        images: List[str] = []
        text_parts: List[str] = []
        for m in message:
            if not isinstance(m, dict) or 'type' not in m:
                continue
            if m['type'] == 'image':
                if 'value' in m and m['value'] is not None:
                    images.append(m['value'])
            elif m['type'] == 'text':
                if 'value' in m and m['value'] is not None:
                    text_parts.append(str(m['value']))
        return images, '\n'.join(text_parts)

    def _render_observer_prompt(self, question: str) -> str:
        from jinja2.sandbox import SandboxedEnvironment

        env = SandboxedEnvironment()
        return env.from_string(self.observer_template).render(content=question)

    def _render_solver_prompt(self, caption: str, question: str) -> str:
        from jinja2.sandbox import SandboxedEnvironment

        env = SandboxedEnvironment()
        rendered = env.from_string(self.solver_template).render(content=question)
        return rendered.replace("<caption>", str(caption), 1)

    def generate_inner(self, message, dataset=None):
        # BaseModel.generate already converts inputs to list[dict(type,value)].
        images, raw_text = self._extract_images_and_text(message)
        disable_clean = os.getenv("DISABLE_QUESTION_CLEAN", "1").lower() in ("1", "true", "yes")
        question_raw = self._normalize_question(
            raw_text,
            strip_question_prefix=not disable_clean,
            collapse_blank_lines=not disable_clean,
        )
        if disable_clean:
            question = question_raw
        else:
            question = self._extract_question_stem(question_raw, dataset)
            question = self._normalize_question(question)

        global _LOG_PROMPT_COUNTER
        _LOG_PROMPT_COUNTER += 1
        log_prompt_env = os.getenv("LOG_PROMPT", "0") == "1"
        do_print = log_prompt_env

        if do_print:
            print("[Question Raw]\n", question_raw, "\n", sep='', flush=True)
            print("[Question Stem]\n", question, "\n", sep='', flush=True)

        # Enforce "images first, single text segment" to match training.
        obs_prompt = self._render_observer_prompt(question)
        obs_msg = [dict(type='image', value=p) for p in images] + [dict(type='text', value=obs_prompt)]
        if do_print:
            print("[Observer Question]\n", question, "\n", sep='', flush=True)
            print("[Observer Prompt]\n", obs_prompt, "\n", sep='', flush=True)
            print("[Observer Message]\n", obs_msg, "\n", sep='', flush=True)
        caption = self.observer_model.generate(message=obs_msg, dataset=dataset).strip()

        sol_prompt = self._render_solver_prompt(caption, question)
        sol_msg = [dict(type='image', value=p) for p in images] + [dict(type='text', value=sol_prompt)]
        if do_print:
            _log_prompt_record(
                {
                    "dataset": dataset,
                    "index": 0,
                    "question_raw": question_raw,
                    "question_processed": question,
                    "observer_prompt": obs_prompt,
                    "solver_prompt": sol_prompt,
                    "images": images,
                }
            )
        if do_print:
            print("[Solver Caption]\n", caption, "\n", sep='', flush=True)
            print("[Solver Prompt]\n", sol_prompt, "\n", sep='', flush=True)
            print("[Solver Message]\n", sol_msg, "\n", sep='', flush=True)
        answer = self.solver_model.generate(message=sol_msg, dataset=dataset)
        if do_print:
            print("[Solver Answer]\n", answer, "\n", sep='', flush=True)

        return answer

    def generate_batch(self, messages, dataset=None):
        """Batched two-stage inference (Observer -> Solver).

        This enables real vLLM concurrency by batching multiple items together at each stage.
        Without this, the outer inference loop calls `generate()` one-by-one and vLLM
        never sees more than one request in flight.
        """

        # Keep behavior consistent with BaseModel.generate: preprocess each message first.
        pp_msgs = []
        for m in messages:
            assert self.check_content(m) in ['str', 'dict', 'liststr', 'listdict'], f'Invalid input type: {m}'
            m = self.preproc_content(m)
            assert m is not None and self.check_content(m) == 'listdict'
            for item in m:
                assert item['type'] in self.allowed_types, f'Invalid input type: {item["type"]}'
            pp_msgs.append(m)

        parsed = []  # (images, question_raw, question_norm)
        print_flags = []
        global _LOG_PROMPT_COUNTER
        log_prompt_env = os.getenv("LOG_PROMPT", "0") == "1"
        for msg in pp_msgs:
            images, question = self._extract_images_and_text(msg)
            question_raw = self._normalize_question(question)
            question = self._extract_question_stem(question_raw, dataset)
            question = self._normalize_question(question)
            parsed.append((images, question_raw, question))
            _LOG_PROMPT_COUNTER += 1
            print_flags.append(log_prompt_env)

        if any(print_flags):
            for i, (pf, (_imgs, question_raw, question_norm)) in enumerate(zip(print_flags, parsed)):
                if not pf:
                    continue
                print(f"[Question Raw #{i}]\n", question_raw, "\n", sep='', flush=True)
                print(f"[Question Stem #{i}]\n", question_norm, "\n", sep='', flush=True)

        # Stage 1: Observer captions (batched)
        obs_msgs = []
        obs_prompts = []
        for images, _question_raw, question in parsed:
            obs_prompt = self._render_observer_prompt(question)
            obs_prompts.append(obs_prompt)
            obs_msg = [dict(type='image', value=p) for p in images] + [dict(type='text', value=obs_prompt)]
            obs_msgs.append(obs_msg)

        if any(print_flags):
            for i, (p, m, pf) in enumerate(zip(obs_prompts, obs_msgs, print_flags)):
                if not pf:
                    continue
                print(f"[Observer Prompt #{i}]\n", p, "\n", sep='', flush=True)
                print(f"[Observer Message #{i}]\n", m, "\n", sep='', flush=True)

        if hasattr(self.observer_model, 'generate_batch'):
            captions = self.observer_model.generate_batch(obs_msgs, dataset=dataset)
        else:
            captions = [self.observer_model.generate(message=m, dataset=dataset) for m in obs_msgs]
        captions = [c.strip() if isinstance(c, str) else str(c) for c in captions]

        # Stage 2: Solver answers (batched)
        sol_msgs = []
        sol_prompts = []
        for (images, question_raw, question), caption in zip(parsed, captions):
            sol_prompt = self._render_solver_prompt(caption, question)
            sol_prompts.append(sol_prompt)
            sol_msg = [dict(type='image', value=p) for p in images] + [dict(type='text', value=sol_prompt)]
            sol_msgs.append(sol_msg)

        if any(print_flags):
            for i, (cap, p, m, pf) in enumerate(zip(captions, sol_prompts, sol_msgs, print_flags)):
                if not pf:
                    continue
                print(f"[Solver Caption #{i}]\n", cap, "\n", sep='', flush=True)
                print(f"[Solver Prompt #{i}]\n", p, "\n", sep='', flush=True)
                print(f"[Solver Message #{i}]\n", m, "\n", sep='', flush=True)

        for i, ((images, question_raw, question), obs_p, sol_p, pf) in enumerate(zip(parsed, obs_prompts, sol_prompts, print_flags)):
            if not pf:
                continue
            _log_prompt_record(
                {
                    "dataset": dataset,
                    "index": i,
                    "question_raw": question_raw,
                    "question_processed": question,
                    "observer_prompt": obs_p,
                    "solver_prompt": sol_p,
                    "images": images,
                }
            )

        if hasattr(self.solver_model, 'generate_batch'):
            answers = self.solver_model.generate_batch(sol_msgs, dataset=dataset)
        else:
            answers = [self.solver_model.generate(message=m, dataset=dataset) for m in sol_msgs]

        if any(print_flags):
            for i, (ans, pf) in enumerate(zip(answers, print_flags)):
                if not pf:
                    continue
                print(f"[Solver Answer #{i}]\n", ans, "\n", sep='', flush=True)

        return answers
