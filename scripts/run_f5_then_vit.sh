#!/bin/bash
# 续跑：ConvNeXt fold 5 (seed=43, LR halved 防 NaN) → ViT 768 单折 → ViT 640 单折
trap '' HUP
set -e
cd /root/autodl-tmp/bs

TIMESTAMP=$(date +%Y%m%d_%H%M)

echo ""
echo "=========================================="
echo "$(date) | ConvNeXt fold 5 retry (seed=43, lr/2)"
echo "=========================================="
python scripts/train_dinov3_multilabel.py \
    --config configs/dinov3_convnext_tiny_multilabel_itksnap_seed43.yaml \
    --run-name dinov3_convnext_tiny_5fold_20260615_0250 \
    --batch-size 12 \
    --grad-accum-steps 1 \
    --learning-rate 5e-5 \
    --backbone-learning-rate 5e-6 \
    --variant tiny \
    --fold f5

echo ""
echo "=========================================="
echo "$(date) | ViT-B/16 single-fold (f1) @ image_size 768"
echo "=========================================="
python scripts/train_dinov3_multilabel.py \
    --config configs/dinov3_vitb16_multilabel_itksnap.yaml \
    --run-name dinov3_vitb16_1fold_768_${TIMESTAMP} \
    --batch-size 8 \
    --grad-accum-steps 1 \
    --fold f1

echo ""
echo "=========================================="
echo "$(date) | ViT-B/16 single-fold (f1) @ image_size 640"
echo "=========================================="
python scripts/train_dinov3_multilabel.py \
    --config configs/dinov3_vitb16_multilabel_itksnap_640.yaml \
    --run-name dinov3_vitb16_1fold_640_${TIMESTAMP} \
    --batch-size 8 \
    --grad-accum-steps 1 \
    --fold f1

echo ""
echo "=========================================="
echo "$(date) | All experiments completed!"
echo "=========================================="
echo "Compare:"
echo "  ConvNeXt 5-fold: runs/dinov3_convnext_tiny_5fold_20260615_0250/fold_summary.csv"
echo "  ViT 768:         runs/dinov3_vitb16_1fold_768_${TIMESTAMP}/f1/train.log"
echo "  ViT 640:         runs/dinov3_vitb16_1fold_640_${TIMESTAMP}/f1/train.log"
