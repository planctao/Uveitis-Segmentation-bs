#!/bin/bash
# compact progress of seg26 runs
cd /root/autodl-tmp/Uveitis-Segmentation-bs
tail -n 3 logs_seg26_queue*.log 2>/dev/null | grep -v '^$'
for d in runs/seg26_*/; do
  for f in $d*/metrics.csv; do
    [ -f "$f" ] || continue
    /root/miniconda3/bin/python - "$f" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
if not rows: sys.exit()
k = 'val_paper_sweep_ind_macro_dice'; e = 'ema_val_paper_sweep_ind_macro_dice'
b = max(rows, key=lambda r: float(r[k]))
s = f"{sys.argv[1].split('/')[1]:32s} ep={len(rows):2d} best={float(b[k]):.4f}@{b['epoch']} (d1={float(b['val_paper_sweep_ind_dice_1']):.4f} d2={float(b['val_paper_sweep_ind_dice_2']):.4f} fix05={float(b['val_paper_macro_dice']):.4f})"
if rows[0].get(e):
    be = max(rows, key=lambda r: float(r[e] or 0))
    s += f" ema={float(be[e]):.4f}@{be['epoch']}"
print(s)
EOF
  done
done
