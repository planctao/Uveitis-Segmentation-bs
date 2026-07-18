#!/usr/bin/env bash
# 并行启动 f1 四组消融（每组一张 A100），后台运行。
# baseline / +DSB / +DALS / +DALS+DSB(full)
set -e
cd "$(dirname "$0")/.."
mkdir -p runs

PY=.venv/bin/python
ITKSNAP=configs/dinov3_convnext_tiny_multilabel_itksnap.yaml
DIFFLEAK=configs/dinov3_convnext_tiny_diffleak.yaml
COMMON="--fold f1 --batch-size 8"

# 卡0: baseline (无 DALS 无 DSB)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src nohup $PY scripts/train_dinov3_multilabel.py \
  --config $ITKSNAP --run-name diffleak_f1_baseline $COMMON \
  > runs/diffleak_f1_baseline.log 2>&1 &
echo "baseline pid=$! -> runs/diffleak_f1_baseline.log"

# 卡1: +DSB only (itksnap 配置无 DALS, CLI 开软边界)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src nohup $PY scripts/train_dinov3_multilabel.py \
  --config $ITKSNAP --run-name diffleak_f1_dsb $COMMON \
  --soft-boundary-sigma 2.0 --soft-boundary-band 7 --soft-boundary-weight 1.0 \
  > runs/diffleak_f1_dsb.log 2>&1 &
echo "dsb pid=$! -> runs/diffleak_f1_dsb.log"

# 卡2: +DALS only (diffleak 配置开 DALS, CLI 关 DSB)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src nohup $PY scripts/train_dinov3_multilabel.py \
  --config $DIFFLEAK --run-name diffleak_f1_dals $COMMON --soft-boundary-sigma 0 \
  > runs/diffleak_f1_dals.log 2>&1 &
echo "dals pid=$! -> runs/diffleak_f1_dals.log"

# 卡3: +DALS+DSB (完整训练侧)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src nohup $PY scripts/train_dinov3_multilabel.py \
  --config $DIFFLEAK --run-name diffleak_f1_full $COMMON \
  > runs/diffleak_f1_full.log 2>&1 &
echo "full pid=$! -> runs/diffleak_f1_full.log"

echo "all 4 ablation jobs launched on GPU 0-3"
