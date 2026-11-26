#!/bin/bash
# Stage compact PhysLeak artefacts into the project (excludes fold_stats.pt caches, dense map npz, mask-derived caches).
set -e
R=/root/autodl-tmp/Uveitis-Segmentation-bs
S=/root/autodl-fs/uveitis_severity_20261013
D=$R/outputs/physleak_severity_20261014
mkdir -p $D
cd $S
for d in physleak physleak_beta_sweep physleak_vitb16 physleak_followup; do
  find $d -type f ! -name fold_stats.pt ! -name dense_maps_frozen.npz ! -name mask_cells_32.npy ! -name cell_cache.npz -print0 \
    | xargs -0 -I{} cp --parents {} $D/
done
mkdir -p $D/logs && cp physleak_sweep.log physleak_vit.log $D/logs/
cat > $D/README.md <<'EOF'
# PhysLeak / CDP compact artefacts (2026-10-14 ~ 10-15)

Compact copy of `/root/autodl-fs/uveitis_severity_20261013/{physleak,physleak_beta_sweep,physleak_vitb16,physleak_followup}`.
Excluded (large / re-derivable): `fold_stats.pt` (feature caches, deleted), `dense_maps_frozen.npz` (90-105 MB per seed, kept only on autodl-fs),
`evaluation/mask_cells_32.npy` and `audit_seed42/cell_cache.npz` (mask-derived evaluation caches).
Target = mask union-area proxy, NOT a clinical grade. See EXPERIMENT_LOG.md section "PhysLeak / CDP" and docs/severity_physleak_*.md.
EOF
du -sh $D; find $D -type f | wc -l; df -h /root/autodl-tmp | tail -1
