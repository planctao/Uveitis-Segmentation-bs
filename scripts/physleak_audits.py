"""Follow-up audits on the frozen PhysLeak beta-sweep outputs (seed 42 by default).

(A) kNN <-> lesion2 confounding audit
(B) CDP disagreement audit (where false-positive cells sit; image residuals; case figure)

Reads frozen predictions/maps + masks (evaluation-only).  No refitting.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fnn
from PIL import Image
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bs.dataset import decode_mask_array  # noqa: E402
from bs.physleak import fov_mask, vesselness, fft_blur  # noqa: E402

SEV = Path('/root/autodl-fs/uveitis_severity_20261013')
G = 32


def rho(a, b): return float(spearmanr(a, b).statistic)


def auc(y, s):
    y = np.asarray(y, bool)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, default=SEV / 'physleak_beta_sweep/seed42')
    p.add_argument('--out', type=Path, default=SEV / 'physleak_followup/audit_seed42')
    p.add_argument('--cdp', default='cdp_phys_beta10')
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    maps = np.load(a.run / 'dense_maps_frozen.npz'); uids = maps['uids']; cm = maps['cellmask']
    frame = pd.read_csv(SEV / 'data/image_groups.csv', dtype={'dhash': str}).set_index('uid')
    tgt = pd.read_csv(SEV / 'data/mask_burden_evaluation_only.csv').set_index('uid')
    pred = pd.read_csv(a.run / 'predictions_frozen.csv'); pred = pred[pred.cond == 'orig']
    fold = frame.loc[uids, 'fold'].to_numpy(); ext = frame.loc[uids, 'extension'].str.lower().to_numpy()
    y = tgt.loc[uids, 'union_fraction'].to_numpy(); l1 = tgt.loc[uids, 'lesion1_fraction'].to_numpy()
    l2 = tgt.loc[uids, 'lesion2_fraction'].to_numpy(); has2 = l2 > 0
    # ---- per-image lesion-specific cell fractions + image appearance descriptors
    cache = a.out / 'cell_cache.npz'
    if cache.exists():
        c = np.load(cache); f1, f2, ves, disc, inten, edge = (c[k] for k in ['f1', 'f2', 'ves', 'disc', 'inten', 'edge'])
    else:
        n = len(uids); f1 = np.zeros((n, G, G), np.float32); f2 = np.zeros_like(f1); ves = np.zeros_like(f1)
        disc = np.zeros((n, G, G), bool); inten = np.zeros_like(f1); edge = np.zeros((n, G, G), bool)
        yy, xx = np.mgrid[:G, :G]
        for i, u in enumerate(uids):
            ip = Path(frame.loc[u, 'image_path'])
            mp = ip.parents[2] / 'mask_only_itksnap' / ip.parent.name / (ip.stem + '.png')
            m = decode_mask_array(np.asarray(Image.open(mp))); h = m.shape[0] // G
            pool = lambda b: b[:h * G, :h * G].reshape(G, h, G, h).mean((1, 3))
            f1[i] = pool(np.isin(m, [1, 3]).astype(np.float32)); f2[i] = pool(np.isin(m, [2, 3]).astype(np.float32))
            im = Image.open(ip).convert('L').resize((512, 512), Image.Resampling.BILINEAR)
            x = torch.from_numpy(np.asarray(im, np.float32) / 255)[None, None]
            fv = fov_mask(x)
            v = vesselness(Fnn.avg_pool2d(x, 2))[None]
            ves[i] = Fnn.adaptive_avg_pool2d(v, G)[0, 0].numpy()
            inten[i] = Fnn.adaptive_avg_pool2d(x, G)[0, 0].numpy()
            bl = fft_blur(x[0] * fv[0].float(), torch.tensor([24.]))[0]
            cy, cx = np.unravel_index(int(bl.argmax()), bl.shape); cy, cx = cy / 16, cx / 16
            disc[i] = (yy + .5 - cy) ** 2 + (xx + .5 - cx) ** 2 <= 3.0 ** 2       # ~96 px diameter at 512
            cmi = torch.from_numpy(cm[i].astype(np.float32))[None, None]
            edge[i] = cm[i] & (-Fnn.max_pool2d(-cmi, 5, 1, 2)[0, 0].numpy() < .5)  # within 2 cells of FOV rim
            if i % 400 == 0: print('cache', i, flush=True)
        np.savez_compressed(cache, f1=f1, f2=f2, ves=ves, disc=disc, inten=inten, edge=edge)
    def img_score(meth, b): return pred[pred.method == meth].set_index('uid').loc[uids, b].to_numpy()
    fov_int = np.array([inten[i][cm[i]].mean() for i in range(len(uids))])
    res = {}
    # =========================== (A) kNN <-> lesion2
    A = []
    cand = {'patch_knn': 'patch_knn', 'cdp': a.cdp}
    for lab, meth in cand.items():
        for b in ['burden_mean', 'burden_pos', 'burden_top10']:
            s = img_score(meth, b)
            row = dict(method=lab, burden=b, l2_auc_all=auc(has2, s), union_rho=rho(s, y))
            for e in sorted(set(ext)): m = ext == e; row[f'l2_auc_{e}'] = auc(has2[m], s[m])
            row['l2_auc_fold_mean'] = float(np.nanmean([auc(has2[fold == f], s[fold == f]) for f in sorted(set(fold))]))
            # within union-area quintile strata (is it more than "bigger lesions")
            qs = np.quantile(y, [.2, .4, .6, .8]); st = np.searchsorted(qs, y)
            row['l2_auc_within_area_quintile_mean'] = float(np.nanmean([auc(has2[st == k], s[st == k]) for k in range(5)]))
            r = rankdata(s); grand = r.mean()
            row['ext_eta2'] = float(sum((ext == e).sum() * (r[ext == e].mean() - grand) ** 2 for e in set(ext)) / ((r - grand) ** 2).sum())
            row['rho_with_fov_intensity'] = rho(s, fov_int)
            A.append(row)
    base = dict(method='fov_mean_intensity', burden='raw', l2_auc_all=auc(has2, fov_int), union_rho=rho(fov_int, y))
    for e in sorted(set(ext)): m = ext == e; base[f'l2_auc_{e}'] = auc(has2[m], fov_int[m])
    A.append(base)
    l2_eta = dict(method='lesion2_presence', burden='label', ext_prev={e: float(has2[ext == e].mean()) for e in sorted(set(ext))})
    A = pd.DataFrame(A); A.to_csv(a.out / 'knn_lesion2_image.csv', index=False)
    # within-image dense: in lesion2-positive images, does kNN rank lesion2 cells above lesion1-only cells?
    D = []
    for lab, meth in cand.items():
        M = maps[f'{meth}__orig'].astype(np.float32)
        a21, a2b, a1b = [], [], []
        for i in np.where(has2)[0]:
            c = cm[i]; s = M[i][c]; c2 = f2[i][c] >= .5; c1 = (f1[i][c] >= .5) & ~c2; c0 = (f1[i][c] + f2[i][c]) == 0
            if c2.sum() >= 2 and c1.sum() >= 2: a21.append(auc(np.r_[np.ones(c2.sum()), np.zeros(c1.sum())], np.r_[s[c2], s[c1]]))
            if c2.sum() >= 2 and c0.sum() >= 2: a2b.append(auc(np.r_[np.ones(c2.sum()), np.zeros(c0.sum())], np.r_[s[c2], s[c0]]))
            if c1.sum() >= 2 and c0.sum() >= 2: a1b.append(auc(np.r_[np.ones(c1.sum()), np.zeros(c0.sum())], np.r_[s[c1], s[c0]]))
        D.append(dict(method=lab, n_img_l2_vs_l1=len(a21), auc_l2cells_vs_l1cells=float(np.mean(a21)),
                      auc_l2cells_vs_bg=float(np.mean(a2b)), auc_l1cells_vs_bg=float(np.mean(a1b))))
        # how much of the image score is carried by lesion cells?  share of top-10% cells that are lesion2 / lesion1 / bg
        share = np.zeros(3); tot = 0
        for i in np.where(has2)[0]:
            c = cm[i]; s = M[i][c]; t = s >= np.quantile(s, .9)
            l2c = f2[i][c][t] > 0; l1c = (f1[i][c][t] > 0) & ~l2c
            share += [l2c.sum(), l1c.sum(), (~l2c & ~l1c).sum()]; tot += t.sum()
        D[-1].update(top10_share_l2=share[0] / tot, top10_share_l1only=share[1] / tot, top10_share_bg=share[2] / tot)
    D = pd.DataFrame(D); D.to_csv(a.out / 'knn_lesion2_dense.csv', index=False)
    res['lesion2_prevalence_by_ext'] = l2_eta['ext_prev']
    # =========================== (B) disagreement audit for CDP
    M = maps[f'{a.cdp}__orig'].astype(np.float32); Mz = M.copy()
    for f in set(fold):
        mm = fold == f; v = M[mm][cm[mm]]; Mz[mm] = (M[mm] - v.mean()) / (v.std() + 1e-9)
    un = np.maximum(f1, f2); pos = un >= .5; neg = un == 0
    thr = np.quantile(Mz[cm & neg], .95)                  # high score = top 5% of mask-negative cells
    fp = cm & neg & (Mz >= thr); tp = cm & pos
    vthr = np.quantile(ves[cm], .8)
    def cats(sel):
        d, e = disc[sel], edge[sel]; v = ves[sel] >= vthr
        c_d = d; c_e = ~d & e; c_v = ~d & ~e & v; c_o = ~d & ~e & ~v
        return dict(n=int(sel.sum()), disc=float(c_d.mean()), fov_edge=float(c_e.mean()), vessel=float(c_v.mean()), other=float(c_o.mean()))
    B = pd.DataFrame([dict(group='all_fov_cells', **cats(cm)), dict(group='mask_positive_cells', **cats(tp)),
                      dict(group='mask_negative_cells', **cats(cm & neg)), dict(group='high_cdp_mask_negative(FP)', **cats(fp))])
    # FP rate per category relative to base rate among mask-negative cells
    rr = {}
    for nm, mk in [('disc', disc), ('fov_edge', edge & ~disc), ('vessel', (ves >= vthr) & ~disc & ~edge)]:
        base_m = cm & neg & mk; rr[nm] = float((fp & mk).sum() / max(base_m.sum(), 1) / .05)
    B.to_csv(a.out / 'cdp_fp_cell_categories.csv', index=False); res['fp_enrichment_vs_5pct'] = rr
    res['disc_cell_mask_positive_rate'] = float(pos[cm & disc].mean()); res['nondisc_mask_positive_rate'] = float(pos[cm & ~disc].mean())
    # image-level residual analysis
    s = img_score(a.cdp, 'burden_pos')
    rs, ry = np.zeros(len(s)), np.zeros(len(s))
    for f in set(fold): m = fold == f; rs[m] = rankdata(s[m]) / m.sum(); ry[m] = rankdata(y[m]) / m.sum()
    resid = rs - ry
    R = dict(resid_by_ext={e: float(resid[ext == e].mean()) for e in sorted(set(ext))},
             resid_rho_fov_intensity=rho(resid, fov_int), resid_rho_l2frac=rho(resid, l2), resid_rho_l1frac=rho(resid, l1),
             resid_mean_l2_present=float(resid[has2].mean()), resid_mean_l2_absent=float(resid[~has2].mean()),
             rho_by_ext={e: rho(s[ext == e], y[ext == e]) for e in sorted(set(ext))},
             zero_area_images=int((y == 0).sum()))
    R['resid_rho_disc_brightness'] = float(pd.Series(resid).corr(pd.Series([inten[i][disc[i] & cm[i]].mean() if (disc[i] & cm[i]).any() else np.nan for i in range(len(s))]), method='spearman'))
    res['image_residuals'] = R
    pd.DataFrame(dict(uid=uids, fold=fold, ext=ext, score_rank=rs, area_rank=ry, resid=resid, union=y, l2=l2, fov_int=fov_int)).to_csv(a.out / 'cdp_image_residuals.csv', index=False)
    # case figure: top-4 over-scored and top-4 under-scored images
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    over, under = np.argsort(-resid)[:4], np.argsort(resid)[:4]
    fig, ax = plt.subplots(8, 3, figsize=(7.5, 20))
    for r, i in enumerate(list(over) + list(under)):
        im = np.asarray(Image.open(frame.loc[uids[i], 'image_path']).convert('L').resize((512, 512)))
        ax[r, 0].imshow(im, cmap='gray'); ax[r, 1].imshow(un[i], cmap='magma', vmin=0, vmax=1)
        mz = np.where(cm[i], Mz[i], np.nan); ax[r, 2].imshow(mz, cmap='magma', vmin=-1, vmax=3)
        ax[r, 2].contour(disc[i], [.5], colors='c', linewidths=.6)
        tag = 'OVER' if r < 4 else 'UNDER'
        ax[r, 0].set_title(f'{tag} {ext[i]} {fold[i]} area={y[i]:.3f}', fontsize=7)
        ax[r, 1].set_title('mask union (cell frac)', fontsize=7); ax[r, 2].set_title(f'CDP z  rank {rs[i]:.2f} vs area {ry[i]:.2f}', fontsize=7)
        for k in range(3): ax[r, k].axis('off')
    plt.tight_layout(); plt.savefig(a.out / 'cdp_disagreement_cases.png', dpi=110)
    (a.out / 'audit_summary.json').write_text(json.dumps(res, indent=2, default=float))
    pd.set_option('display.width', 250)
    print(A.round(3).to_string(index=False)); print(D.round(3).to_string(index=False)); print(B.round(3).to_string(index=False))
    print(json.dumps(res, indent=2, default=float))


if __name__ == '__main__':
    main()
