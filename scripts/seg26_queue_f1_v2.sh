#!/bin/bash
# seg26 f1 screen queue v2 (resumed 2026-10-16 after instance restart). control_s6 treated as done (28/30 ep, best@24).
cd /root/autodl-tmp/Uveitis-Segmentation-bs
PY=/root/miniconda3/bin/python
run(){ name=$1; cfg=$2; shift 2
  if [ -f runs/$name/f1/checkpoints/best.pt ] && grep -q "fold finished" runs/$name/f1/train.log 2>/dev/null; then echo "skip $name"; return; fi
  rm -rf runs/$name
  echo "[$(date +%F' '%T)] start $name"
  $PY scripts/train_dinov3_multilabel.py --config configs/seg_explore_20261015/$cfg.yaml --run-name $name --fold f1 --ema --ema-decay 0.999 --num-workers 16 "$@" > logs_seg26/$name.log 2>&1
  echo "[$(date +%F' '%T)] end $name exit=$?"
}
mkdir -p logs_seg26
run seg26_sample3_f1 sample3
run seg26_sample3_guided_f1 sample3 --logit-upsample guided
run seg26_sample3_dals_f1 sample3_dals
echo QUEUE_DONE
