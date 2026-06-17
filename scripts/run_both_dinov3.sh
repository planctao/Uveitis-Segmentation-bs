#!/bin/bash
# Run DINOv3 ConvNeXt Tiny and ViT-B/16 multilabel experiments sequentially
# This script is meant to be run via: nohup bash scripts/run_both_dinov3.sh &
trap '' HUP  # Ignore hangup signals
set -e
cd /root/autodl-tmp/bs

TIMESTAMP=$(date +%Y%m%d_%H%M)

echo "=========================================="
echo "[$TIMESTAMP] Starting DINOv3 ConvNeXt Tiny 5-fold training"
echo "=========================================="
python scripts/train_dinov3_multilabel.py \
    --config configs/dinov3_convnext_tiny_multilabel_itksnap.yaml \
    --run-name dinov3_convnext_tiny_1fold_${TIMESTAMP} \
    --batch-size 12 \
    --grad-accum-steps 1 \
    --variant tiny \
    --fold f1

echo ""
echo "=========================================="
echo "[$TIMESTAMP] Starting DINOv3 ViT-B/16 5-fold training"
echo "=========================================="
python scripts/train_dinov3_multilabel.py \
    --config configs/dinov3_vitb16_multilabel_itksnap.yaml \
    --run-name dinov3_vitb16_1fold_${TIMESTAMP} \
    --batch-size 4 \
    --grad-accum-steps 1 \
    --fold f1

echo ""
echo "=========================================="
echo "Both experiments completed!"
echo "=========================================="
