#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=false
if [[ $# -eq 1 && "$1" == "--dry-run" ]]; then
    DRY_RUN=true
elif [[ $# -ne 0 ]]; then
    printf 'Usage: %s [--dry-run]\n' "$0" >&2
    exit 2
fi

TRAINING_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT=${DATA_ROOT:-./data}
if [[ "$DATA_ROOT" != /* ]]; then
    DATA_ROOT="$PWD/$DATA_ROOT"
fi
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}
if [[ -d "$MODEL_PATH" ]]; then
    MODEL_PATH=$(cd -- "$MODEL_PATH" && pwd)
fi
TRAIN_JSONL=${TRAIN_JSONL:-"${DATA_ROOT}/train.jsonl"}
VAL_JSONL=${VAL_JSONL:-"${DATA_ROOT}/val.jsonl"}
TRAIN_IMAGE_DIR=${TRAIN_IMAGE_DIR:-"${DATA_ROOT}/train_images"}
VAL_IMAGE_DIR=${VAL_IMAGE_DIR:-"${DATA_ROOT}/val_images"}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/prco_7b}
PYTHON_BIN=${PYTHON_BIN:-python3}

for path_variable in TRAIN_JSONL VAL_JSONL TRAIN_IMAGE_DIR VAL_IMAGE_DIR OUTPUT_DIR; do
    if [[ "${!path_variable}" != /* ]]; then
        printf -v "$path_variable" '%s/%s' "$PWD" "${!path_variable}"
    fi
done

export EXPERIMENT_NAME=prco_7b
export PRCO_LOG_DIR="${OUTPUT_DIR}/logs"
export DRGRPO=1
export caption_rollout_n=4
unset CAPTION_ROLLOUT_N
export use_raw=true
export vllm_judge=true
if [[ "$DRY_RUN" == false ]]; then
    : "${VLLM_JUDGE_IP:?Set VLLM_JUDGE_IP to the judge host}"
    : "${VLLM_JUDGE_PORTS:?Set VLLM_JUDGE_PORTS to comma-separated judge ports}"
fi
export vllm_judge_ip=${VLLM_JUDGE_IP:-}
export vllm_judge_ports=${VLLM_JUDGE_PORTS:-}
export vllm_judge_model=${VLLM_JUDGE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}
export vllm_judge_template="${TRAINING_DIR}/prompts/judge.jinja"
export vllm_judge_use_images=false
unset discard DISCARD vllm_judge_discard VLLM_JUDGE_DISCARD
unset skip_caption_rollout_update SKIP_CAPTION_ROLLOUT_UPDATE use_var_max USE_VAR_MAX
unset vllm_judge_api_key

if [[ "$DRY_RUN" == false ]]; then
    for data_file in "$TRAIN_JSONL" "$VAL_JSONL"; do
        if [[ ! -f "$data_file" ]]; then
            printf 'Dataset file not found: %s\n' "$data_file" >&2
            exit 1
        fi
    done
    for image_directory in "$TRAIN_IMAGE_DIR" "$VAL_IMAGE_DIR"; do
        if [[ ! -d "$image_directory" ]]; then
            printf 'Image directory not found: %s\n' "$image_directory" >&2
            exit 1
        fi
    done
    NUM_GPUS=$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())')
    if [[ "$NUM_GPUS" -eq 0 ]]; then
        printf 'Training requires visible CUDA GPUs. Set CUDA_VISIBLE_DEVICES first.\n' >&2
        exit 1
    fi
else
    NUM_GPUS=8
fi

cd -- "$TRAINING_DIR"
command_args=(
    "$PYTHON_BIN" -m verl.trainer.main
    config=configs/prco_7b.yaml
    "data.train_files=$TRAIN_JSONL"
    "data.val_files=$VAL_JSONL"
    "data.image_dir=$TRAIN_IMAGE_DIR"
    "data.val_image_dir=$VAL_IMAGE_DIR"
    "worker.actor.model.model_path=$MODEL_PATH"
    "trainer.n_gpus_per_node=$NUM_GPUS"
    "trainer.save_checkpoint_path=${OUTPUT_DIR}/checkpoints"
)

if [[ "$DRY_RUN" == true ]]; then
    printf 'Training directory: %s\n' "$TRAINING_DIR"
    printf 'Training steps: 200\n'
    printf 'Command: '
    printf '%q ' "${command_args[@]}"
    printf '\n'
    exit 0
fi

mkdir -p "$PRCO_LOG_DIR"
printf 'Training log: %s/train.log\n' "$PRCO_LOG_DIR"
exec "${command_args[@]}" > "${PRCO_LOG_DIR}/train.log" 2>&1
