#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "$SCRIPT_DIR")"
SCRIPTS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="$SCRIPTS_ROOT/scripts/$SCRIPT_NAME/config.json"
WORKDIR="$SCRIPTS_ROOT/outputs/$SCRIPT_NAME"
export PYTHONUNBUFFERED=1
DEFAULT_CUDA_VISIBLE_DEVICES=${DEFAULT_CUDA_VISIBLE_DEVICES:-0,1}
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    _GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')
  else
    _GPU_COUNT=0
  fi
  if [[ "${_GPU_COUNT}" -ge 2 ]]; then
    export CUDA_VISIBLE_DEVICES="${DEFAULT_CUDA_VISIBLE_DEVICES}"
  else
    export CUDA_VISIBLE_DEVICES=0
  fi
fi
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-64}
export VLLM_BATCH_SIZE=${VLLM_BATCH_SIZE:-64}
export LOG_PROMPT=${LOG_PROMPT:-1}
export LOG_PROMPT_FILE=${LOG_PROMPT_FILE:-$SCRIPT_DIR/check.log}
: > "$LOG_PROMPT_FILE"

LOG_FILE="$SCRIPT_DIR/infer.log"

cd "$SCRIPTS_ROOT"

python "$SCRIPTS_ROOT/run.py" \
  --config "$CONFIG" \
  --reuse \
  --work-dir "$WORKDIR" \
  --mode infer \
  --use-vllm \
  --api-nproc 128 \
  --verbose \
  > "$LOG_FILE" 2>&1
