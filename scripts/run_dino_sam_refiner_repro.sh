#!/usr/bin/env bash
set -euo pipefail
trap '' HUP

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="dinov3_convnext_tiny_vsubr_vw05_repro_${STAMP}"
PIPELINE_DIR="outputs/dino_sam_refiner_repro_${STAMP}"
PIPELINE_LOG="${PIPELINE_DIR}/pipeline.log"
mkdir -p "$PIPELINE_DIR"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

echo "[$(date '+%F %T')] project=$PROJECT_ROOT"
echo "[$(date '+%F %T')] run_name=$RUN_NAME"
echo "[$(date '+%F %T')] starting DINOv3 ConvNeXt-Tiny five-fold training"

for fold in f1 f2 f3 f4 f5; do
  echo "[$(date '+%F %T')] DINO fold=$fold"
  python scripts/train_dinov3_multilabel.py \
    --config configs/dinov3_convnext_tiny_vsubr_vw05.yaml \
    --run-name "$RUN_NAME" \
    --fold "$fold" \
    --batch-size 8 \
    --grad-accum-steps 1 \
    --num-workers 4
done

echo "[$(date '+%F %T')] caching DINO predictions"
python scripts/cache_dino_predictions.py \
  --config configs/dinov3_convnext_tiny_vsubr_vw05.yaml \
  --checkpoint-template "runs/${RUN_NAME}/{fold}/checkpoints/best.pt" \
  --output-root outputs/dino_refiner_cache/vsubr_vw05 \
  --folds f1,f2,f3,f4,f5 \
  --splits train,val \
  --batch-size 4 \
  --num-workers 4

echo "[$(date '+%F %T')] training DINO-SAM interactive refiner"
python scripts/train_interactive_refiner.py \
  --config configs/dino_sam_refiner.yaml \
  --num-workers 4

echo "[$(date '+%F %T')] pipeline completed"
