"""Nested leave-one-fold-out selection of (beta, burden) to quantify optimism of
post-hoc choice: select on 4 folds by mask-area rank rho, evaluate on held-out fold."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
root = Path(sys.argv[1]); SEV = root.parent
tg = pd.read_csv(SEV / 'data/mask_burden_evaluation_only.csv').set_index('uid').union_fraction
out = []
for s in [42, 2026, 3407]:
    p = pd.read_csv(root / f'seed{s}/predictions_frozen.csv')
    p = p[(p.cond == 'orig') & p.method.str.startswith('cdp_phys_beta')]
    long = p.melt(['uid', 'fold', 'method'], ['burden_mean', 'burden_pos', 'burden_top10'], 'burden', 'score')
    long['cfg'] = long.method + '|' + long.burden
    long['y'] = tg.loc[long.uid].to_numpy()
    long['r'] = long.groupby(['cfg', 'fold']).score.rank(pct=True)   # within-fold rank (as in evaluation)
    long['ry'] = long.groupby(['cfg', 'fold']).y.rank(pct=True)
    fr = long.groupby(['cfg', 'fold']).apply(lambda d: spearmanr(d.score, d.y)[0]).unstack()
    held = []
    for f in fr.columns:
        sel = fr.drop(columns=f).mean(1).idxmax()
        held.append(long[(long.cfg == sel) & (long.fold == f)])
        out.append(dict(seed=s, outer=f, selected=sel, inner_rho=fr.drop(columns=f).mean(1).max(), outer_rho=fr.loc[sel, f]))
    h = pd.concat(held)
    out.append(dict(seed=s, outer='POOLED', selected='-', inner_rho=np.nan, outer_rho=spearmanr(h.r, h.y)[0]))
o = pd.DataFrame(out); print(o.round(4).to_string())
print('pooled nested rho per seed:', o[o.outer == 'POOLED'].outer_rho.round(4).tolist(), 'mean', round(o[o.outer == 'POOLED'].outer_rho.mean(), 4))
o.to_csv(root / 'nested_selection.csv', index=False)
