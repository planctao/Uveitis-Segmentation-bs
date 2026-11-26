#!/bin/bash
# Prospective ViT-B/16 test of pre-registered beta rule R1.
set -e
cd /root/autodl-tmp/Uveitis-Segmentation-bs
PY=/root/miniconda3/bin/python
OUT=/root/autodl-fs/uveitis_severity_20261013/physleak_vitb16
export PHYSLEAK_BACKBONE=vitb16
mkdir -p $OUT
sha256sum scripts/physleak_select_beta.py docs/severity_physleak_beta_rule_protocol_20261014.md > $OUT/protocol_sha256_before_run.txt
for s in 42 2026 3407; do
  echo "=== seed $s stats" ; $PY scripts/physleak_stats.py --out $OUT --seed $s --batch 8
  echo "=== seed $s fit"   ; $PY scripts/physleak_fit_predict.py --root $OUT --seed $s --beta-sweep --knn-bank 2000
  echo "=== seed $s rule"  ; $PY scripts/physleak_select_beta.py --run $OUT/seed$s
  echo "=== seed $s eval"  ; $PY scripts/physleak_evaluate.py --root $OUT --seed $s
  rm -f $OUT/seed$s/fold_stats.pt
done
echo VITDONE
