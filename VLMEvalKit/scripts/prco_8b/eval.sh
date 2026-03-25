#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "$SCRIPT_DIR")"
SCRIPTS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="$SCRIPTS_ROOT/scripts/$SCRIPT_NAME/config.json"
WORKDIR="$SCRIPTS_ROOT/outputs/$SCRIPT_NAME"
LOG_FILE="$SCRIPT_DIR/eval.log"

cd "$SCRIPTS_ROOT"

python "$SCRIPTS_ROOT/run.py" \
  --config   "$CONFIG" \
  --work-dir "$WORKDIR" \
  --reuse \
  --mode eval \
  --api-nproc 64 \
  --verbose \
  > "$LOG_FILE" 2>&1
