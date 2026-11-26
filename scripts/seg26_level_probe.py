"""Is the per-image iso-level predictable without GT?  (training-free probe for a 'level head').

For each image/channel: per-threshold counts (inter, pred) and GT size -> per-image oracle level.
Features come only from the probability map (area at each threshold, mean/max, mid-confidence mass).
5-fold CV inside the val fold (images are the unit; no patient IDs -> optimistic upper bound).
Report pooled Dice with: global thr | CV-predicted per-image thr | oracle per-image thr.
"""
import json
import sys
from pathlib import Path

import numpy as np

cache = Path(sys.argv[1]); prob_file = sys.argv[2]
P = np.load(cache / prob_file, mmap_mode="r"); T = np.load(cache / "target.npy", mmap_mode="r"); V = np.load(cache / "valid.npy", mmap_mode="r")
THR = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9])
n = P.shape[0]
res = {}
rng = np.random.default_rng(0)
for c in range(2):
    I = np.zeros((n, len(THR))); S = np.zeros((n, len(THR))); G = np.zeros(n); F = []
    for i in range(n):
        v = V[i, c]; p = P[i, c].astype(np.float32)[v]; t = T[i, c][v]
        G[i] = t.sum()
        for j, th in enumerate(THR):
            pr = p >= th; I[i, j] = (pr & t).sum(); S[i, j] = pr.sum()
        area = S[i] / p.size
        mid = ((p > 0.3) & (p < 0.7)).sum() / max((p > 0.3).sum(), 1)
        hi = p[p > 0.05]
        q = np.quantile(hi, [0.1, 0.25, 0.5, 0.75, 0.9]) if hi.size > 10 else np.zeros(5)
        F.append(np.concatenate([np.log1p(area * 1e4), [mid, p.mean(), p.max()], q]))
    F = np.array(F)
    pooled = lambda idx: 2 * I[np.arange(n), idx].sum() / (S[np.arange(n), idx].sum() + G.sum())
    jg = int((2 * I.sum(0) / (S.sum(0) + G.sum())).argmax())
    # oracle that maximises *pooled* dice greedily is hard; use per-image dice oracle (empty -> min FP)
    orc = np.where(G > 0, (2 * I / (S + G[:, None] + 1e-9)).argmax(1), S.argmin(1))
    # CV: k-NN regression of oracle level on standardized features (simple, low-variance)
    folds = rng.permutation(n) % 5
    pred = np.zeros(n, int)
    for k in range(5):
        tr, te = folds != k, folds == k
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
        A, B = (F[tr] - mu) / sd, (F[te] - mu) / sd
        d = ((B[:, None, :] - A[None]) ** 2).sum(-1)
        nn = np.argsort(d, 1)[:, :15]
        # choose level maximising pooled dice over the neighbours (robust to per-image noise)
        trI, trS, trG = I[tr], S[tr], G[tr]
        for r, row in enumerate(nn):
            dj = 2 * trI[row].sum(0) / (trS[row].sum(0) + trG[row].sum() + 1e-9)
            pred[np.where(te)[0][r]] = int(dj.argmax())
    # shrink toward global level (half-way in index space) as a conservative variant
    shrink = np.round((pred + jg) / 2).astype(int)
    res[f"lesion{c+1}"] = {"global": round(float(pooled(np.full(n, jg))), 4), "cv_knn_level": round(float(pooled(pred)), 4),
                           "cv_knn_shrunk": round(float(pooled(shrink)), 4), "oracle_per_image": round(float(pooled(orc)), 4),
                           "global_thr": float(THR[jg])}
print(json.dumps(res, indent=1))
m = {k: round((res["lesion1"][k] + res["lesion2"][k]) / 2, 4) for k in ["global", "cv_knn_level", "cv_knn_shrunk", "oracle_per_image"]}
print("macro", json.dumps(m))
