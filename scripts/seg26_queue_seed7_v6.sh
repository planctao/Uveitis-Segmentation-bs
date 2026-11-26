#!/bin/bash
# seg26 v6 (2026-10-18): homogeneous-ensemble control for the 5-fold control+PLS result.
# control_s6 seed 7 on f2..f5 (f1 already exists), then TTA eval of control+control7 per fold.
cd /root/autodl-tmp/Uveitis-Segmentation-bs
PY=/root/miniconda3/bin/python
LOG=logs_seg26_queue_f1.log
for fold in f2 f3 f4 f5; do
  name=seg26_control_s6_seed7_$fold
  if [ -f runs/$name/$fold/checkpoints/best.pt ] && grep -q "fold finished" runs/$name/$fold/train.log 2>/dev/null; then
    echo "skip $name" >> $LOG
  else
    rm -rf runs/$name
    echo "[$(date +%F' '%T)] start $name" >> $LOG
    $PY scripts/train_dinov3_multilabel.py --config configs/seg_explore_20261015/control_s6.yaml --run-name $name --fold $fold \
      --seed 7 --num-workers 16 > logs_seg26/$name.log 2>&1
    echo "[$(date +%F' '%T)] end $name exit=$?" >> $LOG
  fi
  C=runs/seg26_control_s6_$fold/$fold/checkpoints/best.pt
  C7=runs/$name/$fold/checkpoints/best.pt
  [ -f $C ] && [ -f $C7 ] && $PY scripts/seg26_eval.py --fold $fold --tta --ckpt $C --ckpt $C7 \
    --output-json outputs/seg26_eval/control+control7_${fold}--tta.json >> logs_seg26/eval_seed7.log 2>&1
  echo "[$(date +%F' '%T)] eval control+control7 $fold done" >> $LOG
done
echo "[$(date +%F' '%T)] V6_SEED7_DONE" >> $LOG
