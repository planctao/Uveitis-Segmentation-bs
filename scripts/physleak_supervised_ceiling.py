"""Supervised reference ceiling for the PhysLeak image-burden protocol.

Out-of-fold HMVF+VS-UBR segmentation (fold-k model on fold-k validation
images, 768 input, sigmoid >= 0.5, no TTA).  Predicted union area is scored
with exactly the same within-fold-rank pooled Spearman protocol as
physleak_evaluate.py.  This is SUPERVISED (trained on other folds' masks) and
is only a reference; f5 checkpoint reached epoch 10 only.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fnn
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bs.inference_runtime import load_model, make_dataset  # noqa: E402
from bs.tta import primary_logits  # noqa: E402

SEV = Path('/root/autodl-fs/uveitis_severity_20261013')
G = 32


def rho(a, b): return float(spearmanr(a, b).statistic)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=SEV / 'physleak_followup/supervised_ceiling')
    p.add_argument('--sweep', type=Path, default=SEV / 'physleak_beta_sweep/seed42')
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(SEV / 'data/image_groups.csv', dtype={'dhash': str})
    frame = frame[~frame.exclude_exact_group]
    key = {(r.fold, Path(r.image_path).stem): r.uid for r in frame.itertuples()}
    rows, cellprob = [], {}
    for k in range(1, 6):
        fold = f'f{k}'
        ck = ROOT / f'runs/hmvf_vsubr_f{k}_30/{fold}/checkpoints/best.pt'
        model, cfg = load_model(ck); model = model.cuda()
        ep = int(torch.load(ck, map_location='cpu', weights_only=True)['epoch'])
        ds = make_dataset(cfg, fold)
        dl = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=4)
        for bt in dl:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                lg = primary_logits(model(bt['image'].cuda()))
            pr = torch.sigmoid(lg.float())                      # (B,2,768,768)
            un_p = pr.amax(1, keepdim=True)
            un_b = (pr >= .5).any(1, keepdim=True).float()
            l2_b = (pr[:, 1:2] >= .5).float()
            cp = Fnn.adaptive_avg_pool2d(un_p, G)[:, 0].cpu().numpy()
            for i, sid in enumerate(bt['sample_id']):
                u = key.get((fold, sid))
                if u is None: continue
                cellprob[u] = cp[i].astype(np.float16)
                rows.append(dict(uid=u, fold=fold, ckpt_epoch=ep,
                                 pred_union_frac=float(un_b[i].mean()), pred_union_soft=float(un_p[i].mean()),
                                 pred_l2_frac=float(l2_b[i].mean()), pred_l2_maxprob=float(pr[i, 1].max())))
        print(fold, 'epoch', ep, len(rows), flush=True)
        del model; torch.cuda.empty_cache()
    pred = pd.DataFrame(rows).drop_duplicates('uid').set_index('uid')
    pred.to_csv(a.out / 'supervised_oof_predictions.csv')
    # ---- score with the PhysLeak protocol
    maps = np.load(a.sweep / 'dense_maps_frozen.npz'); uids = maps['uids']; cm = maps['cellmask']
    cellfrac = np.load(a.sweep / 'evaluation/mask_cells_32.npy')
    tgt = pd.read_csv(SEV / 'data/mask_burden_evaluation_only.csv').set_index('uid')
    fr = frame.set_index('uid')
    assert set(uids) <= set(pred.index), len(set(uids) - set(pred.index))
    y = tgt.loc[uids, 'union_fraction'].to_numpy(); l1 = tgt.loc[uids, 'lesion1_fraction'].to_numpy()
    l2 = tgt.loc[uids, 'lesion2_fraction'].to_numpy(); fold = fr.loc[uids, 'fold'].to_numpy()
    cdp = pd.read_csv(a.sweep / 'predictions_frozen.csv')
    cdp = cdp[(cdp.cond == 'orig') & (cdp.method == 'cdp_phys_beta10')].set_index('uid').loc[uids, 'burden_pos'].to_numpy()
    def wfr(s):
        r = np.zeros(len(s))
        for f in set(fold): m = fold == f; r[m] = rankdata(s[m]) / m.sum()
        return r
    def part(s_, t_, c_):
        rs, rt, rc = rankdata(s_), rankdata(t_), rankdata(c_)
        res = lambda v: v - np.polyval(np.polyfit(rc, v, 1), rc)
        return float(np.corrcoef(res(rs), res(rt))[0, 1])
    rng = np.random.default_rng(20261013); idx = rng.integers(0, len(y), (1000, len(y)))
    def bci(s):
        v = [rho(s[i], y[i]) for i in idx[:300]]
        return np.quantile(v, [.025, .975])
    out = []
    for name, s in [('supervised_union_frac', pred.loc[uids, 'pred_union_frac'].to_numpy()),
                    ('supervised_union_soft', pred.loc[uids, 'pred_union_soft'].to_numpy()),
                    ('cdp_phys_beta10/pos (seed42)', cdp)]:
        rk = wfr(s); ci = bci(rk)
        out.append(dict(method=name, pooled_rank_rho=rho(rk, y), ci_low=ci[0], ci_high=ci[1],
                        **{f'rho_{f}': rho(s[fold == f], y[fold == f]) for f in sorted(set(fold))},
                        lesion1_rho=rho(s, l1), lesion2_presence_auc=float(roc_auc_score(l2 > 0, s)),
                        partial_l2_given_l1=part(s, l2, l1)))
    l2p = pred.loc[uids, 'pred_l2_maxprob'].to_numpy()
    extra = dict(supervised_l2_head_presence_auc=float(roc_auc_score(l2 > 0, l2p)))
    P = np.stack([cellprob[u].astype(np.float32) for u in uids]); pos = cellfrac >= .5
    extra.update(cell_auroc=float(roc_auc_score(pos[cm], P[cm])), cell_ap=float(average_precision_score(pos[cm], P[cm])),
                 cell_prevalence=float(pos[cm].mean()),
                 ckpt_epochs=pred.groupby('fold').ckpt_epoch.first().to_dict(), n=len(uids),
                 note='SUPERVISED out-of-fold reference; f5 checkpoint under-trained; no TTA; threshold 0.5')
    res = pd.DataFrame(out); res.to_csv(a.out / 'ceiling_image_level.csv', index=False)
    (a.out / 'ceiling_summary.json').write_text(json.dumps(extra, indent=2, default=str))
    pd.set_option('display.width', 250); print(res.round(4).to_string(index=False)); print(json.dumps(extra, indent=2, default=str))


if __name__ == '__main__':
    main()
