#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/Uveitis-Segmentation-bs

while tmux has-session -t nnunet_f1 2>/dev/null; do
  sleep 30
done

test -s runs/nnunet2d_clean_f1_resume_final_30/summary.json

python scripts/train_unet_multilabel.py \
  --config configs/aaag_nnunet_2d_multilabel_itksnap.yaml \
  --run-name aaag_nnunet2d_clean_f1_30 \
  --fold f1 --epochs 30 --batch-size 8 --num-workers 8

python scripts/train_unet_multilabel.py \
  --config configs/ukan_nnunet_2d_multilabel_itksnap.yaml \
  --run-name ukan_nnunet2d_clean_f1_30 \
  --fold f1 --epochs 30 --batch-size 8 --num-workers 8

python scripts/train_dinov3_multilabel.py \
  --config configs/dinov3_convnext_tiny_hmvf_vsubr.yaml \
  --run-name hmvf_vsubr_f1_30 \
  --fold f1 --epochs 30 --batch-size 8 --num-workers 8

python scripts/train_dinov3_multilabel.py \
  --config configs/dinov3_convnext_tiny_hmvf_vsubr.yaml \
  --run-name smoke_mcads_vsubr_bs8_2080ti \
  --fold f1 --epochs 1 --batch-size 8 --num-workers 2 \
  --max-train-samples 8 --max-val-samples 8 \
  --head mcads --decoder-attention none

python scripts/train_dinov3_multilabel.py \
  --config configs/dinov3_convnext_tiny_hmvf_vsubr.yaml \
  --run-name mcads_vsubr_f1_30 \
  --fold f1 --epochs 30 --batch-size 8 --num-workers 8 \
  --head mcads --decoder-attention none
