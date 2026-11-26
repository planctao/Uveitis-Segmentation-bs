#!/bin/bash
set -e
cd /root/autodl-tmp/Uveitis-Segmentation-bs/scripts
PY=/root/miniconda3/bin/python
for s in 42 2026 3407; do
  $PY physleak_stats.py --seed $s
  $PY physleak_fit_predict.py --seed $s
  $PY physleak_evaluate.py --seed $s
done
echo ALLDONE
