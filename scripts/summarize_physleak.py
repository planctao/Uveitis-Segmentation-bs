import sys
from pathlib import Path
import pandas as pd
root = Path(sys.argv[1]); seeds = [42, 2026, 3407]
img = pd.concat([pd.read_csv(root / f'seed{s}/evaluation/image_level.csv').assign(seed=s) for s in seeds])
den = pd.concat([pd.read_csv(root / f'seed{s}/evaluation/dense_localisation.csv').assign(seed=s) for s in seeds])
rob = pd.concat([pd.read_csv(root / f'seed{s}/evaluation/gamma_robustness.csv').assign(seed=s) for s in seeds])
cols = ['pooled_rank_rho', 'delta_vs_prior_low', 'delta_vs_prior_high', 'fold_mean_rho', 'fold_sd', 'lesion2_presence_auc', 'partial_l2_given_l1']
g = img.groupby(['method', 'burden'])
t = g[cols].mean().join(g.pooled_rank_rho.std().rename('rho_seed_sd')).join(g.pooled_rank_rho.agg(lambda v: '/'.join(f'{x:.4f}' for x in v)).rename('per_seed'))
pd.set_option('display.width', 300); pd.set_option('display.max_colwidth', 40)
print(t.sort_values('pooled_rank_rho', ascending=False).round(4).to_string())
d = den.groupby('method')[['cell_auroc', 'cell_ap', 'within_image_rho_mean']].agg(['mean', 'std'])
print(d.round(4).to_string())
print(rob.groupby(['method', 'cond'])[['rank_consistency', 'quartile_change']].mean().round(4).to_string())
fold = img[(img.method.str.startswith('cdp_phys_beta3')) | img.method.str.startswith('prior')].groupby(['method', 'burden'])[[c for c in img.columns if c.startswith('rho_f')]].mean()
print(fold.round(4).to_string())
t.to_csv(root / 'summary_image_level.csv'); d.to_csv(root / 'summary_dense.csv')
