#!/bin/bash
# 等待当前 1-fold 训练完成，然后自动启动 5-fold 训练
trap '' HUP
set -e
cd /root/autodl-tmp/bs

MAIN_PID=67418  # 当前 ViT 1-fold 主进程 PID

echo "$(date) | Waiting for ViT 1-fold (PID $MAIN_PID) to finish..."
while kill -0 $MAIN_PID 2>/dev/null; do
    sleep 30
done
echo "$(date) | ViT 1-fold finished. Starting 5-fold experiments..."

sleep 10  # 等待 GPU 显存释放

TIMESTAMP=$(date +%Y%m%d_%H%M)

echo ""
echo "=========================================="
echo "$(date) | Starting ConvNeXt Tiny 5-fold (batch=12)"
echo "=========================================="
python scripts/train_dinov3_multilabel.py \
    --config configs/dinov3_convnext_tiny_multilabel_itksnap.yaml \
    --run-name dinov3_convnext_tiny_5fold_${TIMESTAMP} \
    --batch-size 12 \
    --grad-accum-steps 1 \
    --variant tiny

echo ""
echo "=========================================="
echo "$(date) | Starting ViT-B/16 5-fold (batch=8)"
echo "=========================================="
python scripts/train_dinov3_multilabel.py \
    --config configs/dinov3_vitb16_multilabel_itksnap.yaml \
    --run-name dinov3_vitb16_5fold_${TIMESTAMP} \
    --batch-size 8 \
    --grad-accum-steps 1

echo ""
echo "=========================================="
echo "$(date) | All 5-fold experiments completed!"
echo "=========================================="
