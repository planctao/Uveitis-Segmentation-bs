#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/bs"
UNET_RUN="$PROJECT_ROOT/runs/unet_multilabel_itksnap_512_5fold_50ep_fast_20260614_1833"
WATCH_LOG="$PROJECT_ROOT/runs/dinov3_after_unet_itksnap_watcher.log"

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/runs"

echo "$(date '+%F %T') waiting for U-Net run: $UNET_RUN" | tee -a "$WATCH_LOG"
while screen -ls | grep -q "bs_unet_itksnap_5fold"; do
  sleep 300
  echo "$(date '+%F %T') U-Net screen is still running" | tee -a "$WATCH_LOG"
done

if ! python -c 'import json,sys; p=sys.argv[1]; d=json.load(open(p)); sys.exit(0 if len(d.get("folds", [])) >= 5 else 1)' "$UNET_RUN/summary.json"; then
  echo "$(date '+%F %T') U-Net summary is missing or incomplete; not starting DINOv3." | tee -a "$WATCH_LOG"
  exit 2
fi

STAMP="$(date '+%Y%m%d_%H%M')"
echo "$(date '+%F %T') U-Net complete; probing ConvNeXt-Tiny batch size." | tee -a "$WATCH_LOG"
CONVNEXT_BATCH=8
set +e
python -u scripts/train_dinov3_multilabel.py \
  --config configs/dinov3_convnext_tiny_multilabel_itksnap.yaml \
  --run-name "smoke_dinov3_convnext_tiny_itksnap_batch8_${STAMP}" \
  --fold f5 \
  --epochs 1 \
  --batch-size 8 \
  --max-train-samples 8 \
  --max-val-samples 4 \
  --num-workers 2 \
  2>&1 | tee -a "$WATCH_LOG"
CONVNEXT_PROBE_STATUS=${PIPESTATUS[0]}
set -e
if [ "$CONVNEXT_PROBE_STATUS" -ne 0 ]; then
  CONVNEXT_BATCH=4
  echo "$(date '+%F %T') ConvNeXt-Tiny batch 8 probe failed; falling back to batch 4." | tee -a "$WATCH_LOG"
fi

echo "$(date '+%F %T') starting ConvNeXt-Tiny DINOv3 with batch ${CONVNEXT_BATCH}." | tee -a "$WATCH_LOG"
python -u scripts/train_dinov3_multilabel.py \
  --config configs/dinov3_convnext_tiny_multilabel_itksnap.yaml \
  --run-name "dinov3_convnext_tiny_multilabel_itksnap_768_5fold_30ep_${STAMP}" \
  --batch-size "$CONVNEXT_BATCH" \
  2>&1 | tee -a "$WATCH_LOG"

echo "$(date '+%F %T') ConvNeXt-Tiny complete; probing ViT-B/16 batch size." | tee -a "$WATCH_LOG"
VIT_BATCH=1
VIT_ACCUM=4
set +e
python -u scripts/train_dinov3_multilabel.py \
  --config configs/dinov3_vitb16_multilabel_itksnap.yaml \
  --run-name "smoke_dinov3_vitb16_itksnap_batch2_${STAMP}" \
  --fold f5 \
  --epochs 1 \
  --batch-size 2 \
  --grad-accum-steps 2 \
  --max-train-samples 4 \
  --max-val-samples 2 \
  --num-workers 2 \
  2>&1 | tee -a "$WATCH_LOG"
VIT_PROBE_STATUS=${PIPESTATUS[0]}
set -e
if [ "$VIT_PROBE_STATUS" -eq 0 ]; then
  VIT_BATCH=2
  VIT_ACCUM=2
  echo "$(date '+%F %T') ViT-B/16 batch 2 probe passed." | tee -a "$WATCH_LOG"
else
  echo "$(date '+%F %T') ViT-B/16 batch 2 probe failed; falling back to batch 1." | tee -a "$WATCH_LOG"
fi

echo "$(date '+%F %T') starting ViT-B/16 DINOv3 with batch ${VIT_BATCH}, grad_accum ${VIT_ACCUM}." | tee -a "$WATCH_LOG"
python -u scripts/train_dinov3_multilabel.py \
  --config configs/dinov3_vitb16_multilabel_itksnap.yaml \
  --run-name "dinov3_vitb16_multilabel_itksnap_768_5fold_30ep_${STAMP}" \
  --batch-size "$VIT_BATCH" \
  --grad-accum-steps "$VIT_ACCUM" \
  2>&1 | tee -a "$WATCH_LOG"

echo "$(date '+%F %T') all queued DINOv3 runs finished." | tee -a "$WATCH_LOG"
