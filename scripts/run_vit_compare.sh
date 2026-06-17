#!/bin/bash
# ViT-B/16 单折对比实验：image_size 768 vs 640
trap '' HUP
set -e
cd /root/autodl-tmp/bs

TIMESTAMP=$(date +%Y%m%d_%H%M)

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
echo "$(date) | ViT 768 vs 640 comparison completed!"
echo "=========================================="
echo "Compare:"
echo "  ViT 768: runs/dinov3_vitb16_1fold_768_${TIMESTAMP}/f1/train.log"
echo "  ViT 640: runs/dinov3_vitb16_1fold_640_${TIMESTAMP}/f1/train.log"
