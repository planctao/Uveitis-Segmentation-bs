"""Score the pre-registered R1 forward test on ViT-B/16 (protocol 2026-10-14)."""
import json, sys
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else '/root/autodl-fs/uveitis_severity_20261013/physleak_vitb16')
PRIOR = 0.7065
rows, sel = [], {}
for s in [42, 2026, 3407]:
    d = root / f'seed{s}'
    if not (d / 'evaluation/image_level.csv').exists(): continue
    t = pd.read_csv(d / 'evaluation/image_level.csv'); t['seed'] = s; rows.append(t)
    sel[s] = json.loads((d / 'beta_rule_selection.json').read_text())['selected_method']
img = pd.concat(rows)
cdp = img[img.method.str.startswith('cdp_phys_beta')]
mean = cdp.groupby(['method', 'burden']).pooled_rank_rho.mean().sort_values(ascending=False)
r1 = [float(img[(img.seed == s) & (img.method == m) & (img.burden == 'burden_pos')].pooled_rank_rho.iloc[0]) for s, m in sel.items()]
r1m = sum(r1) / len(r1); oracle = float(mean.iloc[0])
dl = [float(img[(img.seed == s) & (img.method == m) & (img.burden == 'burden_pos')].delta_vs_prior_low.iloc[0]) for s, m in sel.items()]
out = dict(seeds=list(sel), r1_selected=sel, r1_rho_per_seed=r1, r1_rho_mean=r1m, prior=PRIOR,
           oracle_config=list(mean.index[0]), oracle_mean_rho=oracle, gap_to_oracle=oracle - r1m,
           r1_delta_ci_low_per_seed=dl,
           criterion1_ge_prior=r1m >= PRIOR, criterion2_gap_le_0p03=(oracle - r1m) <= .03)
out['success'] = bool(out['criterion1_ge_prior'] and out['criterion2_gap_le_0p03'])
(root / 'r1_forward_test_result.json').write_text(json.dumps(out, indent=2, default=str))
print(json.dumps(out, indent=2, default=str))
print(mean.round(4).to_string())
dense = pd.concat([pd.read_csv(root / f'seed{s}/evaluation/dense_localisation.csv').assign(seed=s) for s in sel])
print(dense.groupby('method')[['cell_auroc', 'within_image_rho_mean']].mean().round(4).to_string())
