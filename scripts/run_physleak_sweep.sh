#!/bin/bash
set -e
cd /root/autodl-tmp/Uveitis-Segmentation-bs/scripts
PY=/root/miniconda3/bin/python
R=/root/autodl-fs/uveitis_severity_20261013
while ! grep -q ALLDONE $R/physleak/run_all.log; do sleep 30; done
for s in 42 2026 3407; do
  mkdir -p $R/physleak_beta_sweep/seed$s
  cp $R/physleak/seed$s/fold_stats.pt $R/physleak/seed$s/stats_manifest.json $R/physleak_beta_sweep/seed$s/
  $PY physleak_fit_predict.py --seed $s --root $R/physleak_beta_sweep --beta-sweep --knn-bank 2000
  $PY physleak_evaluate.py --seed $s --root $R/physleak_beta_sweep
done
echo SWEEPDONE
