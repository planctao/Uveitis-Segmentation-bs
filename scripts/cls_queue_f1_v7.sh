#!/bin/bash
# CLS v7 (2026-10-18): contrast-ratio leakage synthesis vs DALS vs control on f1 (seed 42).
# Order = most informative first. control_s6_f1 (seed42) / seed7_f1 already exist as references.
cd /root/autodl-tmp/Uveitis-Segmentation-bs
PY=/root/miniconda3/bin/python
LOG=logs_seg26_queue_f1.log
run(){ name=$1; cfg=$2; fold=$3; shift 3
  if [ -f runs/$name/$fold/checkpoints/best.pt ] && grep -q "fold finished" runs/$name/$fold/train.log 2>/dev/null; then echo "skip $name" >> $LOG; return; fi
  rm -rf runs/$name
  echo "[$(date +%F' '%T)] start $name" >> $LOG
  $PY scripts/train_dinov3_multilabel.py --config configs/seg_explore_20261018/$cfg.yaml --run-name $name --fold $fold \
    --num-workers 16 "$@" > logs_seg26/$name.log 2>&1
  echo "[$(date +%F' '%T)] end $name exit=$?" >> $LOG
  C=runs/$name/$fold/checkpoints/best.pt
  [ -f $C ] && $PY scripts/seg26_eval.py --fold $fold --tta --ckpt $C --output-json outputs/seg26_eval/${name}--tta.json >> logs_seg26/eval_cls.log 2>&1
  echo "[$(date +%F' '%T)] eval $name done" >> $LOG
}
run seg26_cls_ABC_f1 cls_ABC f1
run seg26_dals_s6_f1 dals_s6 f1
run seg26_cls_A_f1 cls_A f1
run seg26_cls_AB_f1 cls_AB f1
echo "[$(date +%F' '%T)] V7_CLS_F1_DONE" >> $LOG
