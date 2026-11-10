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

import argparse
import os
import re

import numpy as np
import torch
from torch.distributed._tensor import DTensor, Placement, Shard
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForTokenClassification,
    PretrainedConfig,
    PreTrainedModel,
)


def merge_by_placement(tensors: list[torch.Tensor], placement: Placement):
    if placement.is_replicate():
        return tensors[0]
    elif placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet")
    elif placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    else:
        raise ValueError(f"Unsupported placement: {placement}")


def to_bf16(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.bfloat16:
        return tensor
    return tensor.to(dtype=torch.bfloat16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", required=True, type=str, help="The path for your saved model")
    args = parser.parse_args()
    local_dir: str = args.local_dir

    assert not local_dir.endswith("huggingface"), "The local_dir should not end with huggingface."

    # copy rank zero to find the shape of (dp, fsdp)
    rank = 0
    world_size = 0
    for filename in os.listdir(local_dir):
        match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
        if match:
            world_size = match.group(1)
            break

    assert world_size, "No model file with the proper format."

    rank0_weight_path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
    rank0_state_dict = torch.load(rank0_weight_path, map_location="cpu", weights_only=False)
    pivot_key = sorted(rank0_state_dict.keys())[0]
    weight = rank0_state_dict[pivot_key]
    if isinstance(weight, DTensor):
        # get sharding info
        device_mesh = weight.device_mesh
        mesh = device_mesh.mesh
        mesh_dim_names = device_mesh.mesh_dim_names
    else:
        # for non-DTensor
        mesh = np.array([int(world_size)], dtype=np.int64)
        mesh_dim_names = ("fsdp",)

    print(f"Got device mesh {mesh}, mesh_dim_names {mesh_dim_names}")

    assert mesh_dim_names in (("fsdp",), ("ddp", "fsdp")), f"Unsupported mesh_dim_names {mesh_dim_names}."

    if "tp" in mesh_dim_names:
        # fsdp * tp
        total_shards = mesh.shape[-1] * mesh.shape[-2]
        mesh_shape = (mesh.shape[-2], mesh.shape[-1])
    else:
        # fsdp
        total_shards = mesh.shape[-1]
        mesh_shape = (mesh.shape[-1],)

    print(f"Processing {total_shards} model shards in total.")
    keys = list(rank0_state_dict.keys())
    state_dict: dict[str, list[torch.Tensor]] = {key: [] for key in keys}
    param_placements: dict[str, list[Placement]] = {}

    def append_tensor(key: str, tensor: torch.Tensor):
        if isinstance(tensor, DTensor):
            local_tensor = to_bf16(tensor._local_tensor)
            placements = tuple(tensor.placements)
            # replicated placement at ddp dimension can be discarded
            if mesh_dim_names[0] == "ddp":
                placements = placements[1:]

            if key not in param_placements:
                param_placements[key] = placements
            else:
                assert param_placements[key] == placements
        else:
            local_tensor = to_bf16(tensor)

        state_dict[key].append(local_tensor)

    def process_shard(shard_state_dict: dict, shard_rank: int):
        for key in keys:
            try:
                tensor = shard_state_dict.pop(key)
            except KeyError:
                print(f"Cannot find key {key} in rank {shard_rank}.")
                continue
            append_tensor(key, tensor)

    # Process rank0 first to avoid keeping multiple shards in memory.
    process_shard(rank0_state_dict, 0)
    del rank0_state_dict

    # Load remaining shards sequentially to keep peak memory low.
    for shard_rank in range(1, total_shards):
        model_path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{shard_rank}.pt")
        shard_state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        process_shard(shard_state_dict, shard_rank)
        del shard_state_dict

    for key in sorted(state_dict):
        if not isinstance(state_dict[key], list):
            print(f"No need to merge key {key}")
            continue
        if len(state_dict[key]) != total_shards:
            raise ValueError(
                f"Key {key} has {len(state_dict[key])} shards, expected {total_shards}."
            )

        if key in param_placements:
            # merge shards
            placements: tuple[Shard] = param_placements[key]
            if len(mesh_shape) == 1:
                # 1-D list, FSDP without TP
                assert len(placements) == 1
                shards = state_dict[key]
                state_dict[key] = merge_by_placement(shards, placements[0])
            else:
                # 2-D list, FSDP + TP
                raise NotImplementedError("FSDP + TP is not supported yet.")
        else:
            state_dict[key] = torch.cat(state_dict[key], dim=0)

    print("Merge completed.")
    hf_path = os.path.join(local_dir, "huggingface")
    config: PretrainedConfig = AutoConfig.from_pretrained(hf_path)
    architectures: list[str] = getattr(config, "architectures", ["Unknown"])

    if "ForTokenClassification" in architectures[0]:
        AutoClass = AutoModelForTokenClassification
    elif "ForConditionalGeneration" in architectures[0]:
        AutoClass = AutoModelForImageTextToText
    elif "ForCausalLM" in architectures[0]:
        AutoClass = AutoModelForCausalLM
    else:
        raise NotImplementedError(f"Unknown architecture {architectures}.")

    with torch.device("meta"):
        model: PreTrainedModel = AutoClass.from_config(config, torch_dtype=torch.bfloat16)

    assert isinstance(model, PreTrainedModel)

    print(f"Saving model to {hf_path}...")
    model.save_pretrained(hf_path, state_dict=state_dict)
    del state_dict, model
