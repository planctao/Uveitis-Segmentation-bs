#!/bin/bash
# CLS v8 (2026-10-19): cls_B = DALS + ratio appearance only (label 2 only, no co-label, no avoid_foreground).
# Completes the 2x2 factorial {A: co-label+avoid} x {B: ratio appearance}: dals_s6 / cls_A / cls_B / cls_AB.
# Waits for v7 queue to finish (V7_CLS_F1_DONE) so the GPU is never shared.
cd /root/autodl-tmp/Uveitis-Segmentation-bs
PY=/root/miniconda3/bin/python
LOG=logs_seg26_queue_f1.log
echo "[$(date +%F' '%T)] v8 queued (waiting for V7_CLS_F1_DONE)" >> $LOG
until grep -q V7_CLS_F1_DONE $LOG; do sleep 120; done
while pgrep -f train_dinov3_multilabel.py > /dev/null; do sleep 60; done
name=seg26_cls_B_f1; fold=f1
if ! { [ -f runs/$name/$fold/checkpoints/best.pt ] && grep -q "fold finished" runs/$name/$fold/train.log 2>/dev/null; }; then
  rm -rf runs/$name
  echo "[$(date +%F' '%T)] start $name" >> $LOG
  $PY scripts/train_dinov3_multilabel.py --config configs/seg_explore_20261018/cls_B.yaml --run-name $name --fold $fold \
    --num-workers 16 > logs_seg26/$name.log 2>&1
  echo "[$(date +%F' '%T)] end $name exit=$?" >> $LOG
fi
C=runs/$name/$fold/checkpoints/best.pt
[ -f $C ] && $PY scripts/seg26_eval.py --fold $fold --tta --ckpt $C --output-json outputs/seg26_eval/${name}--tta.json >> logs_seg26/eval_cls.log 2>&1
echo "[$(date +%F' '%T)] V8_CLS_B_DONE" >> $LOG
