"""Stage 3: first and only mask read.  Verifies frozen hashes, then evaluates
image-level burden ranking and dense localisation against held-out masks."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bs.dataset import decode_mask_array  # noqa
SEV = Path('/root/autodl-fs/uveitis_severity_20261013')
PRIOR = SEV / 'synthetic_calibration/seed42/predictions_frozen_before_evaluation.csv'


def rho(a, b): return float(spearmanr(a, b).statistic)


def boot_delta(s1, s2, y, n=1000, seed=20261013):
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(y), (n, len(y)))
    def r(s):
        x = rankdata(s[idx], axis=1); x -= x.mean(1, keepdims=True)
        t = rankdata(y[idx], axis=1); t -= t.mean(1, keepdims=True)
        return (x * t).sum(1) / np.sqrt((x * x).sum(1) * (t * t).sum(1))
    a, b = r(s1), r(s2)
    return np.quantile(a, [.025, .975]), np.quantile(a - b, [.025, .975])


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=SEV / 'physleak')
    p.add_argument('--seed', type=int, default=42); a = p.parse_args()
    run = a.root / f'seed{a.seed}'; man = json.loads((run / 'frozen_manifest.json').read_text())
    pp, mp = run / 'predictions_frozen.csv', run / 'dense_maps_frozen.npz'
    assert hashlib.sha256(pp.read_bytes()).hexdigest() == man['predictions_sha256']
    assert hashlib.sha256(mp.read_bytes()).hexdigest() == man['maps_sha256']
    pred = pd.read_csv(pp); maps = np.load(mp); G = maps['cellmask'].shape[-1]
    frame = pd.read_csv(SEV / 'data/image_groups.csv', dtype={'dhash': str}).set_index('uid')
    tgt = pd.read_csv(SEV / 'data/mask_burden_evaluation_only.csv').set_index('uid')
    uids = maps['uids']; out = run / 'evaluation'; out.mkdir(exist_ok=True)
    # dense cell-level targets
    cache = out / f'mask_cells_{G}.npy'
    if cache.exists(): cellfrac = np.load(cache)
    else:
        cellfrac = np.zeros((len(uids), G, G), np.float32)
        for i, u in enumerate(uids):
            ip = Path(frame.loc[u, 'image_path']); mpth = ip.parents[2] / 'mask_only_itksnap' / ip.parent.name / (ip.stem + '.png')
            m = decode_mask_array(np.asarray(Image.open(mpth))); un = np.isin(m, [1, 2, 3]).astype(np.float32)
            h = un.shape[0] // G; cellfrac[i] = un[:h * G, :h * G].reshape(G, h, G, h).mean((1, 3))
        np.save(cache, cellfrac)
    y = tgt.loc[uids, 'union_fraction'].to_numpy(); l2 = tgt.loc[uids, 'lesion2_fraction'].to_numpy()
    l1 = tgt.loc[uids, 'lesion1_fraction'].to_numpy(); fold = frame.loc[uids, 'fold'].to_numpy()
    # prior best label-free method (frozen DINO global + synthetic + gamma projection)
    pr = pd.read_csv(PRIOR); pr = pr[(pr.scope == 'main') & (pr.gamma == 1.) & (pr.method == 'gamma_orthogonal_direction')].set_index('uid')
    prior = pr.loc[uids, 'score'].to_numpy()
    rows = []
    o = pred[pred.cond == 'orig']
    def add(name, burden, s):
        fr = [rho(s[fold == f], y[fold == f]) for f in sorted(set(fold))]
        # within-fold rank -> pooled (transductive, label-free rank normalisation)
        rk = np.zeros_like(s, dtype=float)
        for f in set(fold): m = fold == f; rk[m] = rankdata(s[m]) / m.sum()
        ci, dci = boot_delta(rk, prior, y)
        def part(s_, t_, c_):
            rs, rt, rc = rankdata(s_), rankdata(t_), rankdata(c_)
            res = lambda v: v - np.polyval(np.polyfit(rc, v, 1), rc)
            return float(np.corrcoef(res(rs), res(rt))[0, 1])
        rows.append(dict(method=name, burden=burden, pooled_rank_rho=rho(rk, y), ci_low=ci[0], ci_high=ci[1],
                         delta_vs_prior_low=dci[0], delta_vs_prior_high=dci[1],
                         fold_mean_rho=float(np.mean(fr)), fold_sd=float(np.std(fr, ddof=1)), **{f'rho_{f}': r for f, r in zip(sorted(set(fold)), fr)},
                         lesion1_rho=rho(s, l1), lesion2_presence_auc=float(roc_auc_score(l2 > 0, s)),
                         partial_l2_given_l1=part(s, l2, l1)))
    add('prior_global_gamma_orth (0.7121 ref)', 'score', prior)
    for (n, b) in [(n, b) for n in o.method.unique() for b in ['burden_mean', 'burden_pos', 'burden_top10']]:
        s = o[o.method == n].set_index('uid').loc[uids, b].to_numpy(); add(n, b, s)
    img = pd.DataFrame(rows); img.to_csv(out / 'image_level.csv', index=False)
    # gamma robustness
    rob = []
    for n in pred.method.unique():
        g = pred[pred.method == n]
        if set(g.cond) != {'orig', 'g08', 'g12'}: continue
        s0 = g[g.cond == 'orig'].set_index('uid').loc[uids, 'burden_mean'].to_numpy()
        for c in ['g08', 'g12']:
            s1 = g[g.cond == c].set_index('uid').loc[uids, 'burden_mean'].to_numpy()
            q = np.quantile(s0, [.25, .5, .75])
            rob.append(dict(method=n, cond=c, rank_consistency=rho(s0, s1),
                            quartile_change=float((np.searchsorted(q, s0) != np.searchsorted(q, s1)).mean())))
    pd.DataFrame(rob).to_csv(out / 'gamma_robustness.csv', index=False)
    # dense localisation: cells inside FOV, positive = union fraction>=0.5 in the cell
    cm = maps['cellmask']; pos = (cellfrac >= .5)
    dense = []
    for key in maps.files:
        if not key.endswith('__orig'): continue
        n = key[:-6]; M = maps[key].astype(np.float32)
        Mz = M.copy()
        for f in set(fold):  # per-fold z-score so pooled AUROC is not driven by probe scale
            mm = fold == f; v = M[mm][cm[mm]]; Mz[mm] = (M[mm] - v.mean()) / (v.std() + 1e-9)
        sc, lb = Mz[cm], pos[cm]
        per = [rho(M[i][cm[i]], cellfrac[i][cm[i]]) for i in range(len(uids)) if cellfrac[i][cm[i]].std() > 0 and M[i][cm[i]].std() > 0]
        dense.append(dict(method=n, cell_auroc=float(roc_auc_score(lb, sc)), cell_ap=float(average_precision_score(lb, sc)),
                          prevalence=float(lb.mean()), within_image_rho_mean=float(np.mean(per)), n_images=len(per)))
    pd.DataFrame(dense).to_csv(out / 'dense_localisation.csv', index=False)
    (out / 'evaluation_manifest.json').write_text(json.dumps(dict(
        predictions_sha256=man['predictions_sha256'], n=len(uids), clinical_grading_evaluated=False,
        note='mask-area proxy only; image bootstrap; no patient IDs'), indent=2))
    pd.set_option('display.width', 250)
    print(img[['method', 'burden', 'pooled_rank_rho', 'ci_low', 'ci_high', 'delta_vs_prior_low', 'delta_vs_prior_high', 'fold_mean_rho', 'fold_sd', 'lesion2_presence_auc', 'partial_l2_given_l1']].round(4).to_string(index=False))
    print(pd.DataFrame(rob).round(4).to_string(index=False)); print(pd.DataFrame(dense).round(4).to_string(index=False))


if __name__ == '__main__':
    main()
