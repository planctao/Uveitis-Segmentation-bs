#!/bin/bash
# DINOv3 ViT-B/16 + Fluorescence-Aware MAE Pipeline
# Stage 1: MAE pre-training (fluorescence-weighted masking)
# Stage 2: Segmentation fine-tuning
# Pipeline log → logs/ (git-tracked evidence)
trap '' HUP
set -e
cd /root/autodl-tmp/Uveitis-Segmentation-bs

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M)
FOLD="${1:-f1}"
MAE_EPOCHS="${2:-30}"
FT_EPOCHS="${3:-30}"

echo "=== DINOv3 ViT + Fluorescence MAE Pipeline ==="
echo "Fold: $FOLD | MAE epochs: $MAE_EPOCHS | FT epochs: $FT_EPOCHS"
echo "Start: $(date)"

# Stage 1: MAE pre-training
echo ""
echo "=== [$FOLD] Stage 1: Fluorescence MAE Pre-training ==="
python scripts/train_dinov3_mae.py \
    --stage mae \
    --config configs/dinov3_vitb16_mae_multilabel.yaml \
    --run-name "dinov3_mae_${FOLD}_${TIMESTAMP}" \
    --fold "$FOLD" \
    --epochs "$MAE_EPOCHS"

MAE_CKPT="runs/dinov3_mae_${FOLD}_${TIMESTAMP}/${FOLD}/checkpoints/best.pt"
echo "[$FOLD] MAE checkpoint: $MAE_CKPT"

# Stage 2: Fine-tuning
echo ""
echo "=== [$FOLD] Stage 2: Segmentation Fine-tuning ==="
python scripts/train_dinov3_mae.py \
    --stage finetune \
    --config configs/dinov3_vitb16_mae_multilabel.yaml \
    --run-name "dinov3_mae_ft_${FOLD}_${TIMESTAMP}" \
    --fold "$FOLD" \
    --epochs "$FT_EPOCHS" \
    --mae-ckpt "$MAE_CKPT"

echo ""
echo "=== [$FOLD] Pipeline Complete ==="
echo "End: $(date)"
