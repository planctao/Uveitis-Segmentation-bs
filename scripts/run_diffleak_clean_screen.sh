#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p outputs/logs outputs/runs

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
BASE_CONFIG="configs/dinov3_convnext_tiny_diffleak_cv.yaml"
DALS_CONFIG="configs/dinov3_convnext_tiny_diffleak_dals_cv.yaml"
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
  local config="$3"
  local fold="$4"
  local head="$5"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src "$PYTHON_BIN" scripts/train_dinov3_multilabel.py \
    --config "$config" \
    --run-name "$name" \
    --fold "$fold" \
    --head "$head" \
    --batch-size 8 \
    --grad-accum-steps 1 \
    > "outputs/logs/${name}.log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("$name")
  echo "launched gpu=$gpu pid=$! name=$name"
}

# A second fold checks whether RDH repeats its f1 gain before a full CV run.
launch 0 diffleak_screen_baseline_f2 "$BASE_CONFIG" f2 conv
launch 1 diffleak_screen_rdh_f2 "$BASE_CONFIG" f2 rdh

# Clean-f1 factorial comparison isolates DALS and the RDH x DALS interaction.
launch 2 diffleak_screen_dals_f1 "$DALS_CONFIG" f1 conv
launch 3 diffleak_screen_rdh_dals_f1 "$DALS_CONFIG" f1 rdh

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
