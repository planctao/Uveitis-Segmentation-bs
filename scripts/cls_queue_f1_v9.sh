#!/bin/bash
# v9 (2026-10-19): rerun cls_AB_f1 (v7 attempt OOM at 03:28 because v8 started cls_B concurrently - v8 wait-marker bug).
cd /root/autodl-tmp/Uveitis-Segmentation-bs
PY=/root/miniconda3/bin/python
LOG=logs_seg26_queue_f1.log
while pgrep -f train_dinov3_multilabel.py > /dev/null; do sleep 60; done
name=seg26_cls_AB_f1; fold=f1
rm -rf runs/$name
echo "[$(date +%F' '%T)] start $name (rerun after OOM)" >> $LOG
$PY scripts/train_dinov3_multilabel.py --config configs/seg_explore_20261018/cls_AB.yaml --run-name $name --fold $fold \
  --num-workers 16 > logs_seg26/$name.log 2>&1
echo "[$(date +%F' '%T)] end $name exit=$?" >> $LOG
C=runs/$name/$fold/checkpoints/best.pt
[ -f $C ] && $PY scripts/seg26_eval.py --fold $fold --tta --ckpt $C --output-json outputs/seg26_eval/${name}--tta.json >> logs_seg26/eval_cls.log 2>&1
echo "[$(date +%F' '%T)] V9_RERUN_FINISHED" >> $LOG
