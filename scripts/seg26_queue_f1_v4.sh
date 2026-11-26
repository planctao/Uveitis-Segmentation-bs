#!/bin/bash
# seg26 f1 queue v4 (2026-10-17): after v3 (bash PID 12993) finishes,
# train control seed-7 (2-seed ensemble baseline to disentangle "ensemble gain" vs "PLS gain"),
# then auto-evaluate every finished f1 run with flip-TTA.
cd /root/autodl-tmp/Uveitis-Segmentation-bs
PY=/root/miniconda3/bin/python
LOG=logs_seg26_queue_f1.log
while kill -0 12993 2>/dev/null; do sleep 60; done
name=seg26_control_s6_seed7_f1
rm -rf runs/$name
echo "[$(date +%F' '%T)] start $name" >> $LOG
$PY scripts/train_dinov3_multilabel.py --config configs/seg_explore_20261015/control_s6.yaml --run-name $name --fold f1 \
  --seed 7 --ema --ema-decay 0.999 --num-workers 16 > logs_seg26/$name.log 2>&1
echo "[$(date +%F' '%T)] end $name exit=$?" >> $LOG

E="$PY scripts/seg26_eval.py --fold f1 --tta"
R=runs
for r in seg26_pls_novessel_s6_f1 seg26_sample3_guided_f1 seg26_sample3_dals_f1 seg26_control_s6_seed7_f1; do
  ck=$R/$r/f1/checkpoints
  [ -f $ck/best.pt ] && $E --ckpt $ck/best.pt --ckpt $ck/best_ema.pt --output-json outputs/seg26_eval/${r}--tta.json
done
C0=$R/seg26_control_s6_f1/f1/checkpoints/best.pt
C7=$R/seg26_control_s6_seed7_f1/f1/checkpoints/best.pt
P=$R/seg26_pls_s6_f1/f1/checkpoints/best.pt
[ -f $C7 ] && $E --ckpt $C0 --ckpt $C7 --output-json outputs/seg26_eval/control+control7_f1--tta.json
[ -f $C7 ] && $E --ckpt $C0 --ckpt $C7 --ckpt $P --output-json outputs/seg26_eval/control+control7+pls_f1--tta.json
echo "[$(date +%F' '%T)] V4_EVAL_DONE" >> $LOG
