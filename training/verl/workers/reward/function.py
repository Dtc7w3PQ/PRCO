# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import os
import sys
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Optional, Tuple, TypedDict

import numpy as np

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig
from .vllm_judge import build_judge_from_env, env_flag


class RewardInput(TypedDict):
    response: str
    response_length: int
    ground_truth: str


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[RewardInput], RewardScore]

BatchRewardFunction = Callable[[list[RewardInput]], list[RewardScore]]


class SequentialFunctionRewardManagerMixin:
    reward_fn: SequentialRewardFunction

    def compute_reward_sequential(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            score = self.reward_fn(
                {
                    "response": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                }
            )
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


class BatchFunctionRewardManagerMixin:
    reward_fn: BatchRewardFunction

    def compute_reward_batch(self, caption_batch: DataProto, solve_batch: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_inputs = []
        solve_response_ids = solve_batch.batch["responses"]
        solve_response_length = torch.sum(solve_batch.batch["response_mask"], dim=-1)
        for i in range(len(solve_batch)):
            cur_response_length = int(solve_response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = solve_response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            reward_inputs.append(
                {
                    "response": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": solve_batch.non_tensor_batch["ground_truth"][i],
                }
            )

        reward_metrics = defaultdict(list)

        solve_scores = self.reward_fn(reward_inputs)
        # In self-play, solve_batch contains N solutions per caption.
        # Here we infer the solver group size (solutions per caption).
        n = len(solve_batch) // len(caption_batch)
        if n <= 0:
            raise ValueError(
                f"Invalid self-play batch shapes: len(solve_batch)={len(solve_batch)} "
                f"len(caption_batch)={len(caption_batch)} => n={n}."
            )

        caption_group_n: Optional[int] = None
        cap_raw = os.getenv("caption_rollout_n")
        if cap_raw is None:
            cap_raw = os.getenv("CAPTION_ROLLOUT_N")
        if cap_raw is not None:
            try:
                caption_group_n = int(str(cap_raw).strip())
            except Exception:
                caption_group_n = None

        if caption_group_n is not None:
            if caption_group_n <= 0 or (len(caption_batch) % caption_group_n != 0):
                print(
                    f"[solver_update] WARN invalid caption_rollout_n={caption_group_n}; "
                    f"len(caption_batch)={len(caption_batch)}. Falling back to inference."
                )
                caption_group_n = None

        if caption_group_n is None:
            try:
                puid_arr = caption_batch.non_tensor_batch.get("prompt_uid", None)
                if puid_arr is None:
                    puid_arr = caption_batch.non_tensor_batch.get("uid", None)
                if puid_arr is not None and len(puid_arr) > 0:
                    num_prompts = len(set([str(x) for x in puid_arr]))
                    if num_prompts > 0:
                        inferred = len(caption_batch) // num_prompts
                        if inferred > 0 and inferred * num_prompts == len(caption_batch):
                            caption_group_n = inferred
            except Exception:
                caption_group_n = None

        if caption_group_n is None or caption_group_n <= 0:
            caption_group_n = n

        caption_response_ids = caption_batch.batch["responses"]
        caption_response_length = torch.sum(caption_batch.batch["response_mask"], dim=-1)

        caption_reward_tensor = torch.zeros_like(caption_batch.batch["responses"], dtype=torch.float32)
        caption_reward_key = getattr(self.config, "caption_reward_key", "accuracy")

        # 1) Base caption reward: mean solve quality under each caption.
        caption_scores: list[float] = []
        caption_lens: list[int] = []
        for i in range(len(caption_batch)):
            cur_response_length = int(caption_response_length[i].item())  # avoid tensor indexing error
            caption_lens.append(cur_response_length)

            cur_scores: list[float] = []
            for temp in solve_scores[i * n : (i + 1) * n]:
                v = temp.get(caption_reward_key, None)
                if v is None:
                    v = temp.get("accuracy", None)
                if v is None:
                    v = temp.get("overall", None)
                if v is None:
                    raise KeyError(
                        f"Cannot find caption reward key '{caption_reward_key}' (or fallbacks) "
                        f"in solve_scores entry keys={list(temp.keys())}."
                    )
                cur_scores.append(float(v))

            score = float(sum(cur_scores) / max(len(cur_scores), 1))
            caption_scores.append(score)

        raw_caption_scores = list(caption_scores)

        if env_flag("use_raw", default=False) or env_flag("USE_RAW", default=False):
            reward_metrics["caption_raw"] = [float(x) for x in raw_caption_scores]
            if not getattr(self, "_use_raw_metrics_logged", False):
                print("[reward] use_raw=true; exposing caption_raw for online filtering")
                self._use_raw_metrics_logged = True

        pre_labels = caption_batch.non_tensor_batch.get("vllm_judge_label", None)
        labels: Optional[list[Optional[int]]] = None
        errors: list[str] = []

        if pre_labels is not None:
            try:
                pre_arr = np.asarray(pre_labels, dtype=object)
                labels = pre_arr.reshape(-1).tolist()
            except Exception:
                try:
                    labels = list(pre_labels)
                except Exception:
                    labels = None

            if labels is None or len(labels) != len(caption_batch):
                print(
                    f"[vllm_judge] WARN invalid vllm_judge_label: len(vllm_judge_label)={0 if labels is None else len(labels)} "
                    f"vs len(caption_batch)={len(caption_batch)}. Ignoring precomputed labels and falling back to live judge (if enabled)."
                )
                labels = None

        if labels is None:
            if not hasattr(self, "_vllm_judge_inited"):
                self._vllm_judge_inited = True
                self._vllm_caption_judge = build_judge_from_env()
            judge = getattr(self, "_vllm_caption_judge", None)

            if judge is not None:
                q_key = None
                for cand in ("problem", "prompt", "question", "query", "input", "text"):
                    if cand in caption_batch.non_tensor_batch:
                        q_key = cand
                        break

                obs_responses = caption_batch.batch["responses"]
                obs_lens = torch.sum(caption_batch.batch["response_mask"], dim=-1)

                questions: list[str] = []
                captions: list[str] = []
                mmd_list: list[Any] = []
                mmd_arr = caption_batch.non_tensor_batch.get("multi_modal_data", None)
                q_arr = caption_batch.non_tensor_batch.get(q_key, None) if q_key is not None else None

                for i in range(len(caption_batch)):
                    q = ""
                    if q_arr is not None:
                        try:
                            if len(q_arr) > i:
                                q = str(q_arr[i])
                        except Exception:
                            q = ""

                    cur_len = int(obs_lens[i].item())
                    if cur_len > 0:
                        try:
                            ids = obs_responses[i][:cur_len]
                            if isinstance(ids, torch.Tensor):
                                ids = ids.tolist()
                            cap = self.tokenizer.decode(
                                ids,
                                skip_special_tokens=self.config.skip_special_tokens,
                            )
                        except Exception:
                            cap = ""
                    else:
                        cap = ""

                    mmd = None
                    if mmd_arr is not None:
                        try:
                            if len(mmd_arr) > i:
                                mmd = mmd_arr[i]
                        except Exception:
                            mmd = None

                    questions.append(q)
                    captions.append(cap)
                    mmd_list.append(mmd)

                labels, errors = judge.judge_batch(
                    questions=questions,
                    captions=captions,
                    multi_modal_data_list=mmd_list,
                )

                if labels is not None and len(labels) != len(caption_batch):
                    print(
                        f"[vllm_judge] WARN judge returned {len(labels)} labels for {len(caption_batch)} captions. "
                        "Will align by truncation/padding."
                    )
                    if len(labels) < len(caption_batch):
                        labels = list(labels) + [None] * (len(caption_batch) - len(labels))
                    else:
                        labels = list(labels)[: len(caption_batch)]
        if labels is not None:
            num_zero = 0
            num_fail = 0

            for i, lb in enumerate(labels):
                if lb is None:
                    num_fail += 1
                    reward_metrics["vllm_judge"].append(-1.0)
                    continue

                try:
                    lb_int = int(lb)
                except Exception:
                    lb_int = 1
                reward_metrics["vllm_judge"].append(float(lb_int))

                if lb_int == 0:
                    num_zero += 1
                    caption_scores[i] = 0.0

            if errors:
                print(
                    f"[vllm_judge] Got {len(errors)} judge errors; keeping original caption reward for those. Showing up to {len(errors)} examples:"
                )
                for e in errors:
                    print(f"[vllm_judge]   {e}")

            reward_metrics["vllm_judge_zeroed_frac"] = [float(num_zero) / max(len(labels), 1)]
            reward_metrics["vllm_judge_fail_frac"] = [float(num_fail) / max(len(labels), 1)]

        for i, (score, cur_len) in enumerate(zip(caption_scores, caption_lens)):
            if cur_len > 0:
                caption_reward_tensor[i, cur_len - 1] = float(score)
            reward_metrics["caption"].append(float(score))

        # For GRPO on captions, the learning signal comes from per-prompt differences
        # across captions. If the within-prompt variance is near zero, captions will
        # not receive a useful advantage signal.
        try:
            prompt_uids = caption_batch.non_tensor_batch.get(
                "prompt_uid", caption_batch.non_tensor_batch.get("uid", None)
            )
            if prompt_uids is not None:
                uid2scores = defaultdict(list)
                for uid, sc in zip(prompt_uids, reward_metrics["caption"]):
                    if uid is None:
                        continue
                    uid2scores[str(uid)].append(float(sc))

                if len(uid2scores) > 0:
                    vars_ = [float(np.var(v)) for v in uid2scores.values()]
                    stds_ = [float(np.std(v)) for v in uid2scores.values()]
                    reward_metrics["caption_reward_var"] = vars_
                    reward_metrics["caption_reward_std"] = stds_
                    reward_metrics["caption_reward_var_max"] = [float(np.max(vars_))]
                    reward_metrics["caption_reward_var_min"] = [float(np.min(vars_))]
                    reward_metrics["caption_reward_var_zero_frac"] = [float(np.mean(np.isclose(vars_, 0.0)))]
                else:
                    reward_metrics["caption_reward_var"] = [float(np.var(reward_metrics["caption"]))]
        except Exception:
            pass

        enable_solver_update_sampling = bool(getattr(self.config, "enable_solver_update_sampling", False))
        solver_update_ratio = int(getattr(self.config, "solver_update_ratio", 1))
        enable_group_filter = bool(getattr(self.config, "enable_solver_update_group_filter", True))
        filter_key = str(getattr(self.config, "solver_update_filter_key", "accuracy"))
        enable_random = bool(getattr(self.config, "enable_solver_update_random_sampling", True))
        use_var_max = env_flag("use_var_max", default=False) or env_flag("USE_VAR_MAX", default=False)
        if not getattr(self, "_solver_update_flags_logged", False):
            print(
                "[solver_update] flags: "
                f"enabled={enable_solver_update_sampling} "
                f"ratio={solver_update_ratio} "
                f"group_filter={enable_group_filter} "
                f"filter_key={filter_key} "
                f"random_sampling={enable_random} "
                f"use_var_max={use_var_max}"
            )
            self._solver_update_flags_logged = True

        # per-prompt solve block size = caption_group_n captions * n solutions per caption
        block_size = int(caption_group_n) * int(n)
        if block_size <= 0 or (len(solve_batch) % block_size != 0):
            caption_group_n = n
            block_size = n * n

        idx: list[int] = []
        if not enable_solver_update_sampling:
            for start in range(0, len(solve_batch), block_size):
                idx.extend(range(start, start + n))
        else:
            # We assume the nested order:
            #   per-prompt block size = caption_group_n captions * n solutions = caption_group_n*n.
            # within each block:
            #   caption j's solutions are contiguous at [start + j*n, start + (j+1)*n).
            if solver_update_ratio < 0:
                solver_update_ratio = 0

            for start in range(0, len(solve_batch), block_size):
                candidate_groups = list(range(caption_group_n))  # caption index j

                group_vars = None
                if enable_group_filter or use_var_max:
                    kept_groups: list[int] = []
                    group_vars = []
                    for j in candidate_groups:
                        g0 = start + j * n
                        g1 = g0 + n
                        vals = []
                        for temp in solve_scores[g0:g1]:
                            v = temp.get(filter_key, None)
                            if v is None:
                                v = temp.get("accuracy", None)
                            if v is None:
                                v = temp.get("overall", None)
                            if v is None:
                                vals = None
                                break
                            vals.append(float(v))

                        if vals is None:
                            kept_groups = candidate_groups
                            group_vars = None
                            break

                        var = float(np.var(vals))
                        group_vars.append((var, j))

                        if enable_group_filter:
                            # "all-0" / "all-1" generalization: all-equal group -> no variance -> no GRPO signal.
                            if not (min(vals) == max(vals)):
                                kept_groups.append(j)
                        else:
                            kept_groups.append(j)

                    if enable_group_filter:
                        candidate_groups = kept_groups

                desired_k = min(max(solver_update_ratio, 0), caption_group_n)
                if desired_k == 0:
                    continue

                all_groups = list(range(caption_group_n))
                kept_groups = candidate_groups  # after optional filter

                if use_var_max and group_vars is not None:
                    group_vars_sorted = sorted(group_vars, key=lambda x: x[0], reverse=True)
                    if enable_group_filter:
                        preferred = [j for _, j in group_vars_sorted if j in kept_groups]
                    else:
                        preferred = [j for _, j in group_vars_sorted]
                    chosen = preferred[:desired_k]
                    if len(chosen) < desired_k:
                        remaining = [j for _, j in group_vars_sorted if j not in chosen]
                        chosen.extend(remaining[: (desired_k - len(chosen))])
                elif enable_random:
                    chosen: list[int] = []
                    if len(kept_groups) > 0:
                        k1 = min(desired_k, len(kept_groups))
                        chosen.extend(np.random.choice(kept_groups, size=k1, replace=False).tolist())
                    if len(chosen) < desired_k:
                        need = desired_k - len(chosen)
                        remaining = [g for g in all_groups if g not in chosen]
                        chosen.extend(np.random.choice(remaining, size=need, replace=False).tolist())
                else:
                    chosen = kept_groups[:desired_k]
                    if len(chosen) < desired_k:
                        remaining = [g for g in all_groups if g not in chosen]
                        chosen.extend(remaining[: (desired_k - len(chosen))])

                for j in sorted(chosen):
                    g0 = start + j * n
                    idx.extend(range(g0, g0 + n))

        for i in idx:
            for key, value in solve_scores[i].items():
                reward_metrics[key].append(value)

        solve_reward_tensor = torch.zeros_like(solve_batch.batch["responses"], dtype=torch.float32)
        for i, solve_score in enumerate(solve_scores):
            cur_response_length = int(solve_response_length[i].item())  # avoid tensor indexing error
            solve_reward_tensor[i, cur_response_length - 1] = solve_score["overall"]
        reward_tensor = torch.cat([caption_reward_tensor, solve_reward_tensor[idx]], dim=0)

        caption_batch.batch["role_id"] = torch.zeros(len(caption_batch), dtype=torch.long)
        solve_selected = solve_batch[idx]
        solve_selected.batch["role_id"] = torch.ones(len(solve_selected), dtype=torch.long)

        caption_batch.meta_info["caption_batch_size"] = len(caption_batch)

        # Align non-tensor fields before concatenation: keys that exist only in caption_batch
        # (e.g., a 'split' column in val JSONL) would otherwise break DataProto.concat.
        caption_keys = set(caption_batch.non_tensor_batch.keys())
        solve_keys = set(solve_batch.non_tensor_batch.keys())
        all_keys = caption_keys | solve_keys

        def _ensure_keys(dp: DataProto) -> DataProto:
            for key in all_keys:
                if key not in dp.non_tensor_batch:
                    dp.non_tensor_batch[key] = np.array([None] * len(dp), dtype=object)
            return dp

        batch = DataProto.concat([_ensure_keys(caption_batch), _ensure_keys(solve_selected)])
        return batch, reward_tensor, reward_metrics


class AutoRewardManager(BatchFunctionRewardManagerMixin, SequentialFunctionRewardManagerMixin):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        reward_name = getattr(module, "REWARD_NAME", "unknown")
        reward_type = getattr(module, "REWARD_TYPE", "batch")
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        print(f"Reward name: {reward_name}, reward type: {reward_type}.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.reward_type = reward_type
        self.config = config
        self.tokenizer = tokenizer

    def compute_reward(self, caption_batch: DataProto, solve_batch: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        """Compute reward for a batch of data."""
        if self.reward_type == "batch":
            return self.compute_reward_batch(caption_batch, solve_batch)
        elif self.reward_type == "sequential":
            return self.compute_reward_sequential(caption_batch, solve_batch)
        else:
            raise ValueError(f"Unsupported reward type: {self.reward_type}.")
