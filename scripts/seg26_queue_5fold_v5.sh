#!/bin/bash
# seg26 v5 (2026-10-17): 5-fold confirmation of control vs PLS and control+PLS ensemble (f2..f5; f1 already done).
# Same config/seed as f1 runs; EMA dropped (never beat raw weights on f1).
cd /root/autodl-tmp/Uveitis-Segmentation-bs
PY=/root/miniconda3/bin/python
LOG=logs_seg26_queue_f1.log
run(){ name=$1; fold=$2; shift 2
  if [ -f runs/$name/$fold/checkpoints/best.pt ] && grep -q "fold finished" runs/$name/$fold/train.log 2>/dev/null; then echo "skip $name" >> $LOG; return; fi
  rm -rf runs/$name
  echo "[$(date +%F' '%T)] start $name" >> $LOG
  $PY scripts/train_dinov3_multilabel.py --config configs/seg_explore_20261015/control_s6.yaml --run-name $name --fold $fold \
    --num-workers 16 "$@" > logs_seg26/$name.log 2>&1
  echo "[$(date +%F' '%T)] end $name exit=$?" >> $LOG
}
evalfold(){ fold=$1
  C=runs/seg26_control_s6_$fold/$fold/checkpoints/best.pt
  P=runs/seg26_pls_s6_$fold/$fold/checkpoints/best.pt
  [ -f $C ] && [ -f $P ] || return
  $PY scripts/seg26_eval.py --fold $fold --tta --ckpt $C --ckpt $P --output-json outputs/seg26_eval/control+pls_${fold}--tta.json >> logs_seg26/eval_5fold.log 2>&1
  echo "[$(date +%F' '%T)] eval $fold done" >> $LOG
}
for fold in f2 f3 f4 f5; do
  run seg26_control_s6_$fold $fold
  run seg26_pls_s6_$fold $fold --logit-upsample pls
  evalfold $fold
done
echo "[$(date +%F' '%T)] V5_5FOLD_DONE" >> $LOG
