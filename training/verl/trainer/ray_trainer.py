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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface.
"""

from io import BytesIO
import json
import math
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional, Type, Union
from jinja2 import Template

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin
from PIL import Image
from PIL.Image import Image as ImageObject

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import AutoRewardManager
from ..workers.reward.vllm_judge import build_judge_from_env, env_flag, env_int
from .config import PPOConfig
from .core_algos import (
    AdvantageEstimator,
    FixedKLController,
    KLController,
    compute_advantage_return,
    compute_kl,
    get_kl_controller,
)
from .metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create ray resource pools for distributed training."""
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for different models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards."""
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = torch.mean(VF.masked_mean(kld, mask=response_mask, dim=-1)).item()
    metrics = {"actor/kl_penalty": current_kl, "actor/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    """Compute advantage estimates for policy optimization."""
    adv_inputs = {
        "token_level_rewards": data.batch["token_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "index": data.non_tensor_batch["uid"],
        "gamma": gamma,
        "lam": lam,
    }
    if "values" in data.batch:
        adv_inputs["values"] = data.batch["values"]

    if "reward_baselines" in data.batch:
        adv_inputs["reward_baselines"] = data.batch["reward_baselines"]

    advantages, returns = compute_advantage_return(adv_estimator, **adv_inputs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    # NOTE:
    # Some datasets (e.g. MMK12/ViRL conversions) contain PNGs whose visible content is stored in
    # the alpha channel ("alpha-only RGBA"): RGB channels are (near) all zeros while alpha varies.
    # A naive `image.convert("RGB")` drops the alpha channel and turns these images into pure black,
    # which later gets described by the model as a "black/blank rectangle".
    #
    # Fix: if the image has transparency, composite it over an opaque background (white by default)
    # before converting to RGB.
    if image.mode != "RGB":
        has_transparency = (
            image.mode in {"RGBA", "LA"}
            or (image.mode == "P" and "transparency" in image.info)
            or (image.mode == "RGBa")
        )
        if has_transparency:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            image = image.convert("RGB")

    return image


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[AutoRewardManager] = None,
        val_reward_fn: Optional[AutoRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError

        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        if self.use_reward_model:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()


    def _vllm_judge_discard_enabled(self) -> bool:
        return (
            env_flag("discard", default=False)
            or env_flag("DISCARD", default=False)
            or env_flag("vllm_judge_discard", default=False)
            or env_flag("VLLM_JUDGE_DISCARD", default=False)
        )

    def _get_vllm_caption_judge(self):
        if not hasattr(self, "_vllm_caption_judge_inited"):
            self._vllm_caption_judge_inited = True
            try:
                self._vllm_caption_judge = build_judge_from_env()
            except Exception as e:
                print(f"[vllm_judge] ERROR failed to init caption judge on trainer: {e}")
                self._vllm_caption_judge = None
        return getattr(self, "_vllm_caption_judge", None)

    def _prepare_vllm_judge_inputs(self, batch: DataProto) -> tuple[list[str], list[str], list[Any]]:
        # Question key heuristic: keep identical to reward gating.
        q_key: Optional[str] = None
        for cand in ("problem", "prompt", "question", "query", "input", "text"):
            if cand in batch.non_tensor_batch:
                q_key = cand
                break

        if q_key is None:
            questions = ["" for _ in range(len(batch))]
        else:
            questions = [str(batch.non_tensor_batch[q_key][i]) for i in range(len(batch))]

        # Decode captions from response ids.
        if batch.batch is None or "responses" not in batch.batch:
            captions = ["" for _ in range(len(batch))]
        else:
            response_ids = batch.batch["responses"]
            response_lens = torch.sum(batch.batch["response_mask"], dim=-1)
            captions = []
            for i in range(len(batch)):
                cur_len = int(response_lens[i].item())
                captions.append(self.tokenizer.decode(response_ids[i][:cur_len], skip_special_tokens=True))

        mmd_arr = batch.non_tensor_batch.get("multi_modal_data", None)
        if mmd_arr is None:
            mmd_list = [None for _ in range(len(batch))]
        else:
            mmd_list = [mmd_arr[i] for i in range(len(batch))]

        return questions, captions, mmd_list

    def _maybe_discard_and_resample_captions(
        self,
        caption_prompt_batch: DataProto,
        caption_gen_batch: DataProto,
        caption_batch: DataProto,
        metrics: dict[str, Any],
    ) -> DataProto:
        """Resample captions rejected by the judge before solver generation."""
        if not self._vllm_judge_discard_enabled():
            return caption_batch

        judge = self._get_vllm_caption_judge()
        if judge is None:
            print("[vllm_judge_discard] WARN discard=true but vllm_judge is not available. Skip discard/resample.")
            return caption_batch

        # Caption group size (captions per prompt) for discard/resample.
        group_n = env_int("caption_rollout_n", default=int(self.config.worker.rollout.n))
        group_n = env_int("CAPTION_ROLLOUT_N", default=group_n)
        group_n = max(int(group_n), 1)
        num_prompts = len(caption_prompt_batch)
        if num_prompts <= 0 or group_n <= 0:
            return caption_batch

        if len(caption_batch) != num_prompts * group_n:
            print(
                f"[vllm_judge_discard] WARN unexpected caption_batch size: "
                f"len(caption_batch)={len(caption_batch)} vs num_prompts*n={num_prompts * group_n}. Skip."
            )
            return caption_batch

        def _is_good(lb: Optional[int]) -> bool:
            if lb is None:
                return True
            try:
                return int(lb) != 0
            except Exception:
                return True

        # 1) Judge the initial captions.
        questions, captions, mmd_list = self._prepare_vllm_judge_inputs(caption_batch)
        labels, errors = judge.judge_batch(questions=questions, captions=captions, multi_modal_data_list=mmd_list)
        if errors:
            print(f"[vllm_judge_discard] WARN judge errors (showing up to 4): {errors[:4]}")

        # Per-prompt buckets.
        accepted_parts: list[list[DataProto]] = [[] for _ in range(num_prompts)]
        accepted_labels: list[list[Optional[int]]] = [[] for _ in range(num_prompts)]
        used_initial_local: list[set[int]] = [set() for _ in range(num_prompts)]
        missing: list[int] = [0 for _ in range(num_prompts)]

        num_zero = 0
        num_fail = 0

        for pidx in range(num_prompts):
            start = pidx * group_n
            end = start + group_n
            good_local: list[int] = []
            for j in range(group_n):
                lb = labels[start + j]
                if lb is None:
                    num_fail += 1
                if not _is_good(lb):
                    num_zero += 1
                    continue
                good_local.append(j)

            if good_local:
                idxs = [start + j for j in good_local]
                accepted_parts[pidx].append(caption_batch[idxs])
                accepted_labels[pidx].extend([labels[start + j] for j in good_local])
                used_initial_local[pidx].update(good_local)

            missing[pidx] = max(0, group_n - len(accepted_labels[pidx]))

        # 2) Resample missing captions (batched).
        max_rounds = env_int("vllm_judge_discard_max_rounds", default=20)
        max_rounds = env_int("DISCARD_MAX_ROUNDS", default=max_rounds)
        extra_generated = 0
        rounds_used = 0

        world_size = int(getattr(self.actor_rollout_ref_wg, "world_size", 1))
        world_size = max(world_size, 1)

        for _ in range(max_rounds):
            need_prompts = [i for i in range(num_prompts) if missing[i] > 0]
            if not need_prompts:
                break

            k = max(missing[i] for i in need_prompts)
            k = max(1, min(int(k), group_n))

            # Slice prompt batch and override `n` for this generation call.
            sub_gen = caption_gen_batch[need_prompts]
            sub_gen.meta_info = dict(sub_gen.meta_info)
            sub_gen.meta_info["n"] = k

            sub_gen_padded, pad_size = pad_dataproto_to_divisor(sub_gen, world_size)
            out = self.actor_rollout_ref_wg.generate_sequences(sub_gen_padded)
            out = unpad_dataproto(out, pad_size=pad_size * k)

            extra_generated += len(need_prompts) * k
            rounds_used += 1

            cand = caption_prompt_batch[need_prompts].repeat(repeat_times=k, interleave=True)
            cand = cand.union(out)

            q2, c2, m2 = self._prepare_vllm_judge_inputs(cand)
            labels2, errors2 = judge.judge_batch(questions=q2, captions=c2, multi_modal_data_list=m2)
            if errors2:
                print(f"[vllm_judge_discard] WARN judge errors in resample (showing up to 4): {errors2[:4]}")

            for i_in_need, pidx in enumerate(need_prompts):
                if missing[pidx] <= 0:
                    continue
                start = i_in_need * k
                end = start + k
                for j in range(start, end):
                    lb = labels2[j]
                    if lb is None:
                        num_fail += 1
                    if not _is_good(lb):
                        num_zero += 1
                        continue
                    accepted_parts[pidx].append(cand[j : j + 1])
                    accepted_labels[pidx].append(lb)
                    missing[pidx] -= 1
                    if missing[pidx] <= 0:
                        break

        still_missing = [i for i in range(num_prompts) if missing[i] > 0]
        fallback_added = 0
        fallback_generated = 0
        if still_missing:
            sample = still_missing[:8]
            sample_missing = {i: missing[i] for i in sample}
            print(
                "[vllm_judge_discard] WARN could not fill caption groups after rejection sampling; "
                f"fallback-to-zero-reward for still_missing_prompts={len(still_missing)}/{num_prompts}, "
                f"sample_missing={sample_missing}."
            )

            need_prompts = still_missing
            k = max(missing[i] for i in need_prompts)
            k = max(1, min(int(k), group_n))

            sub_gen = caption_gen_batch[need_prompts]
            sub_gen.meta_info = dict(sub_gen.meta_info)
            sub_gen.meta_info["n"] = k

            sub_gen_padded, pad_size = pad_dataproto_to_divisor(sub_gen, world_size)
            out = self.actor_rollout_ref_wg.generate_sequences(sub_gen_padded)
            out = unpad_dataproto(out, pad_size=pad_size * k)

            fallback_generated = len(need_prompts) * k

            cand = caption_prompt_batch[need_prompts].repeat(repeat_times=k, interleave=True)
            cand = cand.union(out)

            for i_in_need, pidx in enumerate(need_prompts):
                need = int(missing[pidx])
                if need <= 0:
                    continue
                start = i_in_need * k
                accepted_parts[pidx].append(cand[start : start + need])
                accepted_labels[pidx].extend([0 for _ in range(need)])
                fallback_added += need
                missing[pidx] = 0

        if fallback_added > 0:
            metrics["gen/vllm_judge_discard_fallback_added"] = float(fallback_added)
            metrics["gen/vllm_judge_discard_fallback_generated"] = float(fallback_generated)

        # 4) Rebuild the final caption batch (exactly num_prompts * group_n).
        final_batches: list[DataProto] = []
        final_labels_flat: list[Optional[int]] = []
        for pidx in range(num_prompts):
            parts = accepted_parts[pidx]
            if not parts:
                raise RuntimeError(
                    f"[vllm_judge_discard] Internal error: no accepted captions for prompt index={pidx}."
                )

            dp = DataProto.concat(parts)
            if len(dp) != group_n:
                raise RuntimeError(
                    "[vllm_judge_discard] Internal error: unexpected per-prompt batch size. "
                    f"pidx={pidx}, len(dp)={len(dp)}, expected={group_n}."
                )

            cur_labels = accepted_labels[pidx]
            if len(cur_labels) != group_n:
                raise RuntimeError(
                    "[vllm_judge_discard] Internal error: unexpected per-prompt label size. "
                    f"pidx={pidx}, len(labels)={len(cur_labels)}, expected={group_n}."
                )

            final_batches.append(dp)
            final_labels_flat.extend(cur_labels)

        new_caption_batch = DataProto.concat(final_batches)
        new_caption_batch.non_tensor_batch["vllm_judge_label"] = np.array(final_labels_flat, dtype=object)

        total_judged = len(caption_batch) + extra_generated
        if total_judged > 0:
            metrics["gen/vllm_judge_discard_zero_frac"] = num_zero / float(total_judged)
            metrics["gen/vllm_judge_discard_fail_frac"] = num_fail / float(total_judged)
        metrics["gen/vllm_judge_discard_extra_captions"] = float(extra_generated)
        metrics["gen/vllm_judge_discard_rounds"] = float(rounds_used)

        kept = len(new_caption_batch)
        print(
            f"[vllm_judge_discard] caption groups filled: "
            f"num_prompts={num_prompts}, n={group_n}, kept={kept}; "
            f"extra_generated={extra_generated}, rounds={rounds_used}, "
            f"zero={num_zero}, fail={num_fail}"
        )

        return new_caption_batch

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is not None:
            load_checkpoint_path = self.config.trainer.load_checkpoint_path
        elif self.config.trainer.find_last_checkpoint:
            load_checkpoint_path, tracker_info = find_latest_ckpt(self.config.trainer.save_checkpoint_path)
            if tracker_info is not None:
                self.best_val_reward_score = tracker_info.get("best_val_reward_score", 0.0)
                self.best_global_step = tracker_info.get("best_global_step", 0)
        else:
            load_checkpoint_path = None

        if load_checkpoint_path is None:
            return

        if "global_step_" not in load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {load_checkpoint_path}.")
        self.global_step = int(load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _maybe_log_val_generations(
        self,
        inputs: list[str],
        outputs: list[str],
        labels: list[str],
        scores: list[float],
        role_ids: Optional[list[int]] = None,
        prompt_uids: Optional[list[Any]] = None,
    ) -> None:
        """Log validation generations.

        Log one Observer and one Solver generation per prompt example.
        """
        if self.config.trainer.val_generations_to_log <= 0:
            return

        n = int(self.config.trainer.val_generations_to_log)
        samples = list(zip(inputs, outputs, labels, scores))

        if role_ids is None or len(role_ids) != len(samples):
            self.logger.log_generation(samples[:n], self.global_step)
            return

        caption_indices = [i for i, r in enumerate(role_ids) if int(r) == 0]
        solver_indices = [i for i, r in enumerate(role_ids) if int(r) == 1]

        if len(caption_indices) == 0 or len(solver_indices) == 0:
            self.logger.log_generation(samples[:n], self.global_step)
            return

        paired: list[tuple[str, str, str, float]] = []

        # Prefer pairing by prompt_uid (more robust when data is shuffled).
        if prompt_uids is not None and len(prompt_uids) == len(samples):
            solver_by_uid = defaultdict(list)
            for idx in solver_indices:
                solver_by_uid[prompt_uids[idx]].append(idx)

            seen = set()
            for cap_idx in caption_indices:
                uid = prompt_uids[cap_idx]
                if uid in seen:
                    continue
                seen.add(uid)
                sol_list = solver_by_uid.get(uid, [])
                if len(sol_list) == 0:
                    continue
                sol_idx = sol_list[0]

                cap_inp, cap_out, cap_lab, cap_score = samples[cap_idx]
                sol_inp, sol_out, sol_lab, sol_score = samples[sol_idx]

                paired.append((f"[Observer]\n{cap_inp}", cap_out, cap_lab, float(cap_score)))
                paired.append((f"[Solver]\n{sol_inp}", sol_out, sol_lab, float(sol_score)))

                if len(paired) >= 2 * n:
                    break
        else:
            for cap_idx, sol_idx in zip(caption_indices, solver_indices):
                cap_inp, cap_out, cap_lab, cap_score = samples[cap_idx]
                sol_inp, sol_out, sol_lab, sol_score = samples[sol_idx]

                paired.append((f"[Observer]\n{cap_inp}", cap_out, cap_lab, float(cap_score)))
                paired.append((f"[Solver]\n{sol_inp}", sol_out, sol_lab, float(sol_score)))

                if len(paired) >= 2 * n:
                    break

        if len(paired) == 0:
            self.logger.log_generation(samples[:n], self.global_step)
            return

        self.logger.log_generation(paired, self.global_step)

    def _validate(self) -> dict[str, Any]:
        reward_tensor_lst = []
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        sample_role_ids, sample_prompt_uids = [], []
        reward_metrics_lst = defaultdict(list)
        length_metrics_lst = defaultdict(list)
        print("Start validation...")
        use_images = self.config.data.solve_use_images
        switch_step = getattr(self.config.data, "solve_use_images_switch_step", -1)
        if (not use_images) and int(switch_step) >= 0 and self.global_step >= int(switch_step):
            use_images = True
            if not getattr(self, "_solve_use_images_switched", False):
                print(f"[RayPPOTrainer] solve_use_images switched to True at global_step={self.global_step} (switch_step={switch_step})")
                self._solve_use_images_switched = True
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            caption_test_batch = DataProto.from_single_dict(batch_dict)
            if 'uid' not in caption_test_batch.non_tensor_batch:
                caption_test_batch.non_tensor_batch['uid'] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(caption_test_batch))], dtype=object
                )
            if 'prompt_uid' not in caption_test_batch.non_tensor_batch:
                caption_test_batch.non_tensor_batch['prompt_uid'] = caption_test_batch.non_tensor_batch['uid']
            test_gen_batch = caption_test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = dict(self.config.worker.rollout.val_override_config)  # avoid side effects
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps
            if self.config.data.caption_max_tokens is not None:
                caption_max_tokens = int(self.config.data.caption_max_tokens)
                if caption_max_tokens > self.config.data.max_response_length:
                    raise ValueError(
                        f"data.caption_max_tokens ({caption_max_tokens}) must be <= data.max_response_length "
                        f"({self.config.data.max_response_length})."
                    )
                test_gen_batch.meta_info["max_tokens"] = caption_max_tokens

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            # repeat to align with repeated responses in rollout
            caption_test_batch = caption_test_batch.repeat(repeat_times=repeat_times, interleave=True)
            caption_test_batch = caption_test_batch.union(test_output_gen_batch)

            prompt_uid_arr = caption_test_batch.non_tensor_batch.get(
                'prompt_uid', caption_test_batch.non_tensor_batch.get('uid', None)
            )

            tensors = defaultdict(list)
            non_tensors = defaultdict(list)
            for i in range(len(caption_test_batch)):
                problem = caption_test_batch.non_tensor_batch[self.config.data.prompt_key][i]
                caption = self.tokenizer.decode(caption_test_batch.batch['responses'][i], skip_special_tokens=True)
                caption = caption.replace("<image>", "[image]")
                with open(self.config.data.solve_prompt, encoding="utf-8") as f:
                    solve_prompt = Template(f.read().strip()).render(content=problem)
                if use_images:
                    solve_prompt = "<image>" + solve_prompt.replace("<image>", "")
                else:
                    solve_prompt = solve_prompt.replace("<image>", "")
                solve_prompt = solve_prompt.replace("<caption>", caption)

                content_list = []
                for j, content in enumerate(solve_prompt.split("<image>")):
                    if j != 0 and use_images:
                        content_list.append({"type": "image"})

                    if content:
                        content_list.append({"type": "text", "text": content})

                messages = [{"role": "user", "content": content_list}]

                solve_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                processed_images = caption_test_batch.non_tensor_batch['multi_modal_data'][i]['processed_images']
                if not use_images:
                    processed_images = None
                model_inputs = self.processor(processed_images, [solve_prompt], add_special_tokens=False, return_tensors="pt")
                input_ids = model_inputs.pop("input_ids")[0]
                attention_mask = model_inputs.pop("attention_mask")[0]

                if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
                    # qwen-vl mrope
                    if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                        from ..models.transformers.qwen3_vl import get_rope_index
                    else:
                        from ..models.transformers.qwen2_vl import get_rope_index

                    vision_position_ids = get_rope_index(
                        self.processor,
                        input_ids=input_ids,
                        image_grid_thw=model_inputs.get("image_grid_thw", None),
                        video_grid_thw=model_inputs.get("video_grid_thw", None),
                        second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                        attention_mask=attention_mask,
                    )  # (3, seq_length)
                    text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
                    position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
                else:
                    position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

                input_ids, attention_mask, position_ids = VF.postprocess_data(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    max_length=self.config.data.max_prompt_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=True,
                    truncation=self.config.data.truncation,
                )
                raw_prompt_ids = self.tokenizer.encode(solve_prompt, add_special_tokens=False)
                if len(raw_prompt_ids) > self.config.data.max_prompt_length:
                    if self.config.data.truncation == "left":
                        raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
                    elif self.config.data.truncation == "right":
                        raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
                    elif self.config.data.truncation == "error":
                        raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")
                tensors["input_ids"].append(input_ids)
                tensors["attention_mask"].append(attention_mask)
                tensors["position_ids"].append(position_ids)
                non_tensors["raw_prompt_ids"].append(raw_prompt_ids)
                non_tensors["multi_modal_data"].append(caption_test_batch.non_tensor_batch['multi_modal_data'][i])
                non_tensors["problem"].append(problem)
                non_tensors["ground_truth"].append(caption_test_batch.non_tensor_batch['ground_truth'][i])
                non_tensors["prompt_uid"].append(prompt_uid_arr[i] if prompt_uid_arr is not None else None)
            for key, value in tensors.items():
                tensors[key] = torch.stack(value, dim=0)
            for key, value in non_tensors.items():
                non_tensors[key] = np.array(value, dtype=object)
            solve_batch_dict = {**tensors, **non_tensors}
            solve_test_batch: DataProto = DataProto.from_single_dict(solve_batch_dict)
            test_gen_batch = solve_test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            if not use_images:
                # drop images completely to avoid vllm expecting image placeholders
                test_gen_batch.non_tensor_batch.pop("multi_modal_data", None)
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = dict(self.config.worker.rollout.val_override_config)  # avoid caption max_tokens leak
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps
            # Ensure solve generation isn't truncated by caption max_tokens.
            test_gen_batch.meta_info.pop("max_tokens", None)

            solver_max_tokens = test_gen_batch.meta_info.get("max_tokens", None)
            if not getattr(self, "_printed_solver_max_tokens", False):
                print(
                    "[DEBUG] Validation solver max_tokens=",
                    solver_max_tokens,
                    "(None means rollout default); val_override_config=",
                    self.config.worker.rollout.val_override_config,
                    "max_response_length=",
                    self.config.data.max_response_length,
                )
                self._printed_solver_max_tokens = True

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            # repeat to align with repeated responses in rollout
            solve_test_batch = solve_test_batch.repeat(repeat_times=repeat_times, interleave=True)
            solve_test_batch = solve_test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            test_batch, reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(caption_test_batch, solve_test_batch))

            # store generations
            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)
            # Track role / prompt ids for paired generation logging.
            if 'role_id' in test_batch.batch:
                sample_role_ids.extend(test_batch.batch['role_id'].detach().cpu().tolist())
            if 'prompt_uid' in test_batch.non_tensor_batch:
                sample_prompt_uids.extend(test_batch.non_tensor_batch['prompt_uid'].tolist())

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

            for key, value in compute_length_metrics(test_batch).items():
                length_metrics_lst[key].append(value)

        self.actor_rollout_ref_wg.release_rollout_engine()
        self._maybe_log_val_generations(
            sample_inputs,
            sample_outputs,
            sample_labels,
            sample_scores,
            role_ids=sample_role_ids if len(sample_role_ids) == len(sample_inputs) else None,
            prompt_uids=sample_prompt_uids if len(sample_prompt_uids) == len(sample_inputs) else None,
        )
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        val_length_metrics = {f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}
        print("Finish validation.")
        return {**val_reward_metrics, **val_length_metrics}

    def _balance_batch(self, batch: DataProto, metrics: dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _make_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        try:
            batch_dict = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.train_dataloader)
            batch_dict = next(self.data_iterator)

        meta_info = {
            "min_pixels": self.config.data.min_pixels,
            "max_pixels": self.config.data.max_pixels,
            "video_fps": self.config.data.video_fps,
        }

        print("Start generating caption...")
        caption_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
        caption_batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(caption_batch.batch))], dtype=object
        )
        # Keep a stable prompt-level uid for filtering / bookkeeping.
        # - caption_batch.non_tensor_batch["uid"] is used as group id for caption GRPO.
        # - solve_batch will overwrite its own "uid" to represent caption-level groups;
        #   therefore we store the prompt uid separately as "prompt_uid".
        caption_batch.non_tensor_batch["prompt_uid"] = caption_batch.non_tensor_batch["uid"]
        use_images = self.config.data.solve_use_images
        switch_step = getattr(self.config.data, "solve_use_images_switch_step", -1)
        if (not use_images) and int(switch_step) >= 0 and self.global_step >= int(switch_step):
            use_images = True
            if not getattr(self, "_solve_use_images_switched", False):
                print(f"[RayPPOTrainer] solve_use_images switched to True at global_step={self.global_step} (switch_step={switch_step})")
                self._solve_use_images_switched = True

        # pop those keys for generation
        gen_batch = caption_batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
        )
        if self.config.data.caption_max_tokens is not None:
            caption_max_tokens = int(self.config.data.caption_max_tokens)
            if caption_max_tokens > self.config.data.max_response_length:
                raise ValueError(
                    f"data.caption_max_tokens ({caption_max_tokens}) must be <= data.max_response_length "
                    f"({self.config.data.max_response_length})."
                )
            gen_batch.meta_info["max_tokens"] = caption_max_tokens

        # Caption GRPO group size (captions per prompt). This can be smaller than solver rollout.n.
        # Override via env vars: `caption_rollout_n` / `CAPTION_ROLLOUT_N`.
        caption_rollout_n = env_int("caption_rollout_n", default=int(self.config.worker.rollout.n))
        caption_rollout_n = env_int("CAPTION_ROLLOUT_N", default=caption_rollout_n)
        caption_rollout_n = max(int(caption_rollout_n), 1)

        # vLLM sampling params can be overridden per-call via meta_info.
        # See: vLLMRollout.update_sampling_params().
        gen_batch.meta_info["n"] = caption_rollout_n

        if not getattr(self, "_selfplay_group_sizes_logged", False):
            print(
                "[selfplay] group sizes: "
                f"caption_rollout_n={caption_rollout_n} "
                f"solver_rollout_n={int(self.config.worker.rollout.n)}"
            )
            self._selfplay_group_sizes_logged = True

        # generate a batch
        gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)
        # repeat to align with repeated responses in rollout
        caption_prompt_batch = caption_batch
        caption_batch = caption_prompt_batch.repeat(repeat_times=caption_rollout_n, interleave=True)
        caption_batch = caption_batch.union(gen_batch_output)

        caption_batch = self._maybe_discard_and_resample_captions(
            caption_prompt_batch=caption_prompt_batch,
            caption_gen_batch=gen_batch,
            caption_batch=caption_batch,
            metrics=metrics,
        )

        print(f"Finish generating.")

        print("Start generating solution...")
        tensors = defaultdict(list)
        non_tensors = defaultdict(list)
        for i in range(len(caption_batch)):
            problem = caption_batch.non_tensor_batch[self.config.data.prompt_key][i]
            caption = self.tokenizer.decode(caption_batch.batch['responses'][i], skip_special_tokens=True)
            caption = caption.replace("<image>", "[image]")
            with open(self.config.data.solve_prompt, encoding="utf-8") as f:
                solve_prompt = Template(f.read().strip()).render(content=problem)
            if use_images:
                solve_prompt = "<image>" + solve_prompt.replace("<image>", "")
            else:
                solve_prompt = solve_prompt.replace("<image>", "")
            solve_prompt = solve_prompt.replace("<caption>", caption)
            solve_prompt_raw = solve_prompt

            content_list = []
            for j, content in enumerate(solve_prompt.split("<image>")):
                if j != 0 and use_images:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            messages = [{"role": "user", "content": content_list}]

            solve_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            processed_images = caption_batch.non_tensor_batch['multi_modal_data'][i]['processed_images']
            if not use_images:
                processed_images = None
            model_inputs = self.processor(processed_images, [solve_prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

            if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
                # qwen-vl mrope
                if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                    from ..models.transformers.qwen3_vl import get_rope_index
                else:
                    from ..models.transformers.qwen2_vl import get_rope_index

                vision_position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids,
                    image_grid_thw=model_inputs.get("image_grid_thw", None),
                    video_grid_thw=model_inputs.get("video_grid_thw", None),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                    attention_mask=attention_mask,
                )  # (3, seq_length)
                text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
                position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
            else:
                position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

            try:
                input_ids, attention_mask, position_ids = VF.postprocess_data(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    max_length=self.config.data.max_prompt_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=True,
                    truncation=self.config.data.truncation,
                )
            except RuntimeError as e:
                msg = str(e)
                if "Input sequence length" in msg and "max length" in msg:
                    uid = caption_batch.non_tensor_batch["uid"][i]
                    debug_dir = os.path.join(self.config.trainer.save_checkpoint_path, "debug_long_prompt")
                    os.makedirs(debug_dir, exist_ok=True)
                    debug_path = os.path.join(
                        debug_dir,
                        (
                            "solve_prompt_uid="
                            f"{uid}_i={i}_seqlen={len(input_ids)}_max={self.config.data.max_prompt_length}.txt"
                        ),
                    )
                    try:
                        with open(debug_path, "w", encoding="utf-8") as f:
                            f.write(f"uid={uid}\n")
                            f.write(f"i={i}\n")
                            f.write(f"seq_len(input_ids)={len(input_ids)}\n")
                            f.write(f"max_prompt_length={self.config.data.max_prompt_length}\n")
                            f.write(f"error={msg}\n")
                            f.write("\n===== problem =====\n")
                            f.write(str(problem))
                            f.write("\n\n===== caption =====\n")
                            f.write(str(caption))
                            f.write("\n\n===== solve_prompt_raw(with <image>) =====\n")
                            f.write(str(solve_prompt_raw))
                            f.write("\n\n===== solve_prompt_chat_template =====\n")
                            f.write(str(solve_prompt))
                    except Exception as write_exc:
                        print(f"[PromptTooLong] Failed to write debug prompt to {debug_path}: {write_exc}")
                    else:
                        print(f"[PromptTooLong] Wrote debug prompt to: {debug_path}")
                raise
            raw_prompt_ids = self.tokenizer.encode(solve_prompt, add_special_tokens=False)
            if len(raw_prompt_ids) > self.config.data.max_prompt_length:
                if self.config.data.truncation == "left":
                    raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
                elif self.config.data.truncation == "right":
                    raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
                elif self.config.data.truncation == "error":
                    raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")
            tensors["input_ids"].append(input_ids)
            tensors["attention_mask"].append(attention_mask)
            tensors["position_ids"].append(position_ids)
            non_tensors["raw_prompt_ids"].append(raw_prompt_ids)
            non_tensors["multi_modal_data"].append(caption_batch.non_tensor_batch['multi_modal_data'][i])
            non_tensors["problem"].append(problem)
            non_tensors["ground_truth"].append(caption_batch.non_tensor_batch['ground_truth'][i])
            # propagate prompt-level uid for downstream filtering / analysis
            non_tensors["prompt_uid"].append(caption_batch.non_tensor_batch["uid"][i])
        for key, value in tensors.items():
            tensors[key] = torch.stack(value, dim=0)
        for key, value in non_tensors.items():
            non_tensors[key] = np.array(value, dtype=object)
        solve_batch_dict = {**tensors, **non_tensors}
        solve_batch: DataProto = DataProto.from_single_dict(solve_batch_dict, meta_info=meta_info)
        solve_batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(solve_batch.batch))], dtype=object
        )
        # pop those keys for generation
        gen_batch = solve_batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
        )
        if not use_images:
            # drop images completely to avoid vllm expecting image placeholders
            gen_batch.non_tensor_batch.pop("multi_modal_data", None)

        # generate a batch
        gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)

        # repeat to align with repeated responses in rollout
        solve_batch = solve_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
        solve_batch = solve_batch.union(gen_batch_output)
        print(f"Finish generating.")

        return caption_batch, solve_batch

    def _make_batch_data_with_online_filtering(self, metrics: dict[str, Any]) -> DataProto:
        """Generate a training batch with DAPO-style online filtering.

        Groups without variation in caption reward are filtered out.
        """
        batch: Optional[DataProto] = None
        num_try_make_batch = 0
        print("Start generating batch with online filtering...")
        if not getattr(self, "_online_filtering_logged", False):
            print(
                "[online_filtering] enabled; "
                f"filter_low={self.config.algorithm.filter_low} "
                f"filter_high={self.config.algorithm.filter_high} "
                f"rollout_batch_size={self.config.data.rollout_batch_size}"
            )
            self._online_filtering_logged = True

        while True:
            num_try_make_batch += 1
            caption_batch, solve_batch = self._make_batch_data(metrics=metrics)

            # Compute reward and build the merged batch (caption + selected solver samples).
            new_batch, reward_tensor, reward_metrics = ray.get(
                self.reward_fn.compute_reward.remote(caption_batch, solve_batch)
            )
            new_batch.batch["token_level_scores"] = reward_tensor

            # Filter prompt groups using caption part scores.
            caption_size = int(new_batch.meta_info.get("caption_batch_size", 0))
            if caption_size <= 0:
                if "role_id" in new_batch.batch:
                    caption_size = int((new_batch.batch["role_id"] == 0).sum().item())
                else:
                    caption_size = len(new_batch) // 2

            caption_scores = None
            used_raw = False
            use_raw = env_flag("use_raw", default=False) or env_flag("USE_RAW", default=False)
            if not getattr(self, "_online_filter_use_raw_logged", False):
                print(f"[online_filtering] use_raw env={use_raw} (caption_raw preferred when available)")
                self._online_filter_use_raw_logged = True
            if use_raw and isinstance(reward_metrics, dict):
                raw_scores = reward_metrics.get("caption_raw", None)
                if raw_scores is not None:
                    try:
                        raw_scores = list(raw_scores)
                    except Exception:
                        raw_scores = None

                if raw_scores is not None and len(raw_scores) >= caption_size:
                    try:
                        caption_scores = [float(x) for x in raw_scores[:caption_size]]
                        used_raw = True
                    except Exception:
                        caption_scores = None
                else:
                    if raw_scores is None:
                        print(
                            "[online_filtering] use_raw=true but reward_metrics has no 'caption_raw'; "
                            "falling back to gated caption scores."
                        )
                    else:
                        print(
                            f"[online_filtering] use_raw=true but len(caption_raw)={len(raw_scores)} "
                            f"!= caption_size={caption_size}; falling back to gated caption scores."
                        )
                    caption_scores = None

            if caption_scores is None:
                # Use caption sequence score (= last-token reward; stored token-level with only last token non-zero).
                caption_scores = new_batch.batch["token_level_scores"][:caption_size].sum(-1).cpu().numpy().tolist()
            if used_raw and not getattr(self, "_use_raw_logged", False):
                print("[online_filtering] use_raw=true; using caption_raw for filtering (ok)")
                self._use_raw_logged = True
            caption_uids = new_batch.non_tensor_batch["uid"][:caption_size]

            uid2scores = defaultdict(list)
            for uid, score in zip(caption_uids, caption_scores):
                uid2scores[uid].append(score)

            kept_uids = [
                uid
                for uid, scores in uid2scores.items()
                if min(scores) != max(scores)
            ]

            if len(kept_uids) == 0:
                raise RuntimeError(
                    "No prompt is kept after online filtering. "
                    "Please check your data / reward or relax filter_low/filter_high."
                )

            prompt_uid_arr = new_batch.non_tensor_batch.get("prompt_uid", new_batch.non_tensor_batch["uid"])
            kept_sample_idxs = [i for i, puid in enumerate(prompt_uid_arr) if puid in kept_uids]
            new_batch = new_batch[kept_sample_idxs]

            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch

            # Count how many *prompt groups* we have collected so far.
            if "role_id" in batch.batch:
                caption_mask = (batch.batch["role_id"] == 0).cpu().numpy().astype(bool)
                caption_uid_arr = batch.non_tensor_batch["uid"][caption_mask]
            else:
                caption_uid_arr = batch.non_tensor_batch["uid"][: int(len(batch) // 2)]

            seen = set()
            ordered_prompt_uids = []
            for uid in caption_uid_arr:
                if uid not in seen:
                    seen.add(uid)
                    ordered_prompt_uids.append(uid)

            current_prompt_count = len(ordered_prompt_uids)
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_prompt_count < rollout_batch_size:
                print(f"{current_prompt_count=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                    continue
                raise RuntimeError(
                    f"{num_try_make_batch=} >= {max_try_make_batch=}. "
                    "Generated too many batches for online filtering. Please check your data."
                )

            # Truncate to the first `rollout_batch_size` prompt groups while keeping all samples for those prompts.
            selected_uids = set(ordered_prompt_uids[:rollout_batch_size])
            prompt_uid_arr = batch.non_tensor_batch.get("prompt_uid", batch.non_tensor_batch["uid"])
            final_idxs = [i for i, puid in enumerate(prompt_uid_arr) if puid in selected_uids]
            batch = batch[final_idxs]

            # Refresh caption_batch_size meta for correct logging/slicing downstream.
            if "role_id" in batch.batch:
                batch.meta_info["caption_batch_size"] = int((batch.batch["role_id"] == 0).sum().item())
            else:
                batch.meta_info["caption_batch_size"] = int(len(batch) // 2)

            if isinstance(reward_metrics, dict) and len(reward_metrics) > 0:
                metrics.update({f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()})

            print(f"{current_prompt_count=} >= {rollout_batch_size=}. Finish generating.")
            return batch

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[dict[str, Any]] = None

        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    if self.config.algorithm.online_filtering:
                        batch = self._make_batch_data_with_online_filtering(metrics=metrics)
                    else:
                        caption_batch, solve_batch = self._make_batch_data(metrics=metrics)
                    self.actor_rollout_ref_wg.release_rollout_engine()

                with timer("reward", timing_raw):
                    if not self.config.algorithm.online_filtering:
                        batch, reward_tensor, reward_metrics = ray.get(
                            self.reward_fn.compute_reward.remote(caption_batch, solve_batch)
                        )
                        batch.batch["token_level_scores"] = reward_tensor
                        reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                        metrics.update(reward_metrics)

                if env_flag("skip_caption_rollout_update", default=False) or env_flag(
                    "SKIP_CAPTION_ROLLOUT_UPDATE", default=False
                ):
                    if not getattr(self, "_skip_caption_rollout_update_logged", False):
                        print("[ablation] skip_caption_rollout_update=true; dropping caption samples before updates")
                        self._skip_caption_rollout_update_logged = True

                    if "role_id" in batch.batch:
                        keep = (batch.batch["role_id"] == 1)
                        keep_idx = torch.nonzero(keep, as_tuple=False).view(-1)
                        batch = batch[keep_idx]
                    else:
                        caption_size = int(batch.meta_info.get("caption_batch_size", len(batch) // 2))
                        caption_size = max(min(caption_size, len(batch)), 0)
                        batch = batch[caption_size:]

                    # Refresh meta for downstream logging/slicing.
                    batch.meta_info["caption_batch_size"] = 0

                with timer("old", timing_raw):
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                    batch = batch.union(old_log_probs)

                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                        batch = batch.union(ref_log_probs)

                if self.use_critic:
                    with timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with timer("adv", timing_raw):
                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                    )

                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)

                    critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                    metrics.update(critic_metrics)

                if self.config.trainer.critic_warmup <= self.global_step:
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)

                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    metrics.update(actor_metrics)

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()

                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
