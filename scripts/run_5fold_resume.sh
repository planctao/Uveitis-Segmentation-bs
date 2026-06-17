#!/bin/bash
# 5-fold 训练 - 优化 checkpoint 保存（save_interval=999 已生效）
# ConvNeXt fold 1 已完成（best_paper_macro_dice=0.7612），从 fold 2 继续
trap '' HUP
set -e
cd /root/autodl-tmp/bs

TIMESTAMP=$(date +%Y%m%d_%H%M)

# 复用之前已完成的 ConvNeXt fold1 结果
CONVNEXT_RUN_DIR="runs/dinov3_convnext_tiny_5fold_20260615_0250"

echo ""
echo "=========================================="
echo "$(date) | Continue ConvNeXt 5-fold: f2,f3,f4,f5"
echo "=========================================="
for FOLD in f3 f4 f5; do
    echo ""
    echo ">>> ConvNeXt fold $FOLD <<<"
    python scripts/train_dinov3_multilabel.py \
        --config configs/dinov3_convnext_tiny_multilabel_itksnap.yaml \
        --run-name dinov3_convnext_tiny_5fold_20260615_0250 \
        --batch-size 12 \
        --grad-accum-steps 1 \
        --variant tiny \
        --fold $FOLD
done

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
echo "$(date) | ConvNeXt 5-fold + ViT 1-fold (768 vs 640) completed!"
echo "=========================================="
echo "Compare results:"
echo "  ViT 768: runs/dinov3_vitb16_1fold_768_${TIMESTAMP}/f1/train.log"
echo "  ViT 640: runs/dinov3_vitb16_1fold_640_${TIMESTAMP}/f1/train.log"
