#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p outputs/logs outputs/runs

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
BASE_CONFIG="configs/dinov3_convnext_tiny_diffleak_cv.yaml"
DMT_CONFIG="configs/dinov3_convnext_tiny_dmt_dice2_cv.yaml"
FOLD="${FOLD:-f1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PIDS=()
NAMES=()

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup INT TERM

launch() {
  local gpu="$1"
  local name="$2"
  shift 2
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src "$PYTHON_BIN" scripts/train_dinov3_multilabel.py \
    --run-name "$name" \
    --fold "$FOLD" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum-steps 1 \
    "$@" \
    > "outputs/logs/${name}.log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("$name")
  echo "launched gpu=$gpu pid=$! name=$name"
}

launch 0 "dmt_screen_baseline_${FOLD}" --config "$BASE_CONFIG" --head conv
launch 1 "dmt_screen_boundary_dou_${FOLD}" --config "$BASE_CONFIG" --head conv --boundary-dou-weight 0.3
launch 2 "dmt_screen_dice2_${FOLD}" --config "$DMT_CONFIG" --head conv --dmt-close-weight 0.0
launch 3 "dmt_screen_dice2_close_${FOLD}" --config "$DMT_CONFIG" --head conv --dmt-close-weight 0.05

status=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    echo "completed name=${NAMES[$index]}"
  else
    code=$?
    echo "failed code=$code name=${NAMES[$index]}" >&2
    status=1
  fi
done
exit "$status"
