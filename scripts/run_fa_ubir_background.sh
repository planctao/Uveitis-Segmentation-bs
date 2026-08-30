#!/usr/bin/env bash
set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_NAME="fa_ubir_v1"
OUTPUT_DIR="outputs/interactive_boundary_runs/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR"
LOG_PATH="$OUTPUT_DIR/background.log"

exec nohup python -u scripts/train_interactive_boundary.py \
  --config configs/fa_ubir_v1.yaml \
  > "$LOG_PATH" 2>&1 &
