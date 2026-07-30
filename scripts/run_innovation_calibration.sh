#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p outputs/evaluation outputs/logs

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CONFIG="configs/dinov3_convnext_tiny_diffleak.yaml"
PIDS=()
NAMES=()

launch() {
  local gpu="$1"
  local name="$2"
  local checkpoint="$3"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src "$PYTHON_BIN" scripts/evaluate_dinov3_postprocess.py \
    --config "$CONFIG" \
    --checkpoint "$checkpoint" \
    --fold f1 \
    --batch-size 8 \
    --num-workers 8 \
    --ablation-suite \
    --disable-tta \
    --disable-postprocess \
    --disable-intensity-refine \
    --disable-fov-mask \
    --output-json "outputs/evaluation/${name}.json" \
    > "outputs/logs/${name}_calibration.log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("$name")
  echo "launched gpu=$gpu pid=$! name=$name"
}

launch 0 rdh_v1_f1 runs/diffleak_f1_rdh_clean/f1/checkpoints/best.pt
launch 1 rdh_dals_f1 outputs/runs/diffleak_screen_rdh_dals_f1/f1/checkpoints/best.pt
launch 2 rdh_v2_f1 outputs/runs/diffleak_screen_rdhv2_f1/f1/checkpoints/best.pt
launch 3 dals_f1 outputs/runs/diffleak_screen_dals_f1/f1/checkpoints/best.pt

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
