"""Stage 2 (GPU, mask-free): fit fold-wise Counterfactual Difference Probes from
stage-1 statistics, select ridge strength by inner synthetic cross-validation
(label-free), then produce dense leakage-potential maps and image burden scores
for held-out folds.  Also computes label-free dense baselines.  Predictions are
frozen (sha256) before any mask is read by the evaluation script.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fnn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'backbone/dinov3'))
from bs.physleak import dense_features, cells, fov_mask, quantize, solve_cdp, sse, GRID  # noqa
from physleak_stats import DS, load_model, SEV  # noqa

LAMS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]
# name -> (generator kinds, nuisance beta)   main = 'cdp_phys'  (fixed before evaluation)
VARIANTS = {'cdp_phys': (('phys',), 1.0), 'cdp_gauss': (('gauss',), 1.0), 'cdp_both': (('phys', 'gauss'), 1.0),
            'cdp_phys_nonuis': (('phys',), 0.0), 'cdp_phys_beta3': (('phys',), 3.0), 'cdp_phys_beta03': (('phys',), 0.3)}


def agg(st, ks, kinds, beta):
    A = sum(st[f'A_{t}'][ks].sum(0) / st[f'n_{t}'][ks].sum() for t in kinds)
    b = sum(st[f'b_{t}'][ks].sum(0) / st[f'n_{t}'][ks].sum() for t in kinds)
    yy = sum(st[f'yy_{t}'][ks].sum() / st[f'n_{t}'][ks].sum() for t in kinds)
    N = st['N_mm'][ks].sum(0) / st['n_mm'][ks].sum() + st['N_gamma'][ks].sum(0) / st['n_gamma'][ks].sum()
    n = st['n_cells'][ks].sum(); mu = st['S1'][ks].sum(0) / n
    sd = (st['S2'][ks].sum(0).diagonal() / n - mu ** 2).clamp_min(1e-12).sqrt()
    return A, b, yy, N * beta, mu, sd


def fit_variant(st, train_ks, kinds, beta):
    # inner leave-one-training-fold-out on *synthetic* SSE only (no masks)
    A, b, yy, N, mu, sd = agg(st, train_ks, kinds, beta)
    cv = {}
    for lam in LAMS:
        e = 0.
        for v in train_ks:
            inner = [k for k in train_ks if k != v]
            Ai, bi, _, Ni, _, sdi = agg(st, inner, kinds, beta)
            w = solve_cdp(Ai, bi, Ni, 1.0, lam, sdi)
            Av, bv, yyv, _, _, _ = agg(st, [v], kinds, beta)
            e += float(sse(w, Av, bv, yyv, sdi)) / float(yyv)
        cv[lam] = e / len(train_ks)
    lam = min(cv, key=cv.get)
    w = solve_cdp(A, b, N, 1.0, lam, sd)
    r2 = 1 - float(sse(w, A, b, yy, sd)) / float(yy)
    return dict(w=(w / sd), mu=mu, lam=lam, cv=cv, train_synth_r2=r2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=SEV / 'physleak')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--knn-bank', type=int, default=60000)
    p.add_argument('--beta-sweep', action='store_true', help='post-hoc sensitivity: phys generator, beta grid')
    a = p.parse_args()
    global VARIANTS
    if a.beta_sweep:
        VARIANTS = {f'cdp_phys_beta{b:g}': (('phys',), b) for b in [0, 0.3, 1, 3, 10, 30, 100]}
    dev = 'cuda'; run = a.root / f'seed{a.seed}'
    man = json.loads((run / 'stats_manifest.json').read_text())
    st = {k: v.to(dev) for k, v in torch.load(run / 'fold_stats.pt', weights_only=True).items()}
    folds = man['folds']; K = len(folds)
    probes = {}
    for name, (kinds, beta) in VARIANTS.items():
        for k in range(K):
            tr = [j for j in range(K) if j != k]
            probes[(name, k)] = fit_variant(st, tr, kinds, beta)
            print(name, folds[k], 'lam', probes[(name, k)]['lam'], 'synthR2 %.3f' % probes[(name, k)]['train_synth_r2'], flush=True)
    frame = pd.read_csv(SEV / 'data/image_groups.csv', dtype={'dhash': str})
    frame = frame[~frame.exclude_exact_group].reset_index(drop=True)
    frame = frame.set_index('uid').loc[man['uids']].reset_index()
    assert len(frame) == man['n_images']
    fidx = frame.fold.map({f: i for i, f in enumerate(folds)}).to_numpy()
    model, _ = load_model(dev)
    dl = torch.utils.data.DataLoader(DS(frame.image_path.tolist(), man['size']), batch_size=a.batch, num_workers=6)
    conds = {'orig': lambda x: x, 'g08': lambda x: quantize(x ** .8), 'g12': lambda x: quantize(x ** 1.2)}
    N = len(frame); G = GRID
    maps = {(n, c): np.zeros((N, G, G), np.float16) for n in VARIANTS for c in conds}
    maps.update({(n, c): np.zeros((N, G, G), np.float16) for n in ['intensity', 'dog_prior'] for c in conds})
    cellmask = np.zeros((N, G, G), bool)
    feats = None
    t0 = time.monotonic()
    for x, idx in dl:
        x = x.to(dev); idx = idx.numpy(); k = fidx[idx]
        cm = cells(fov_mask(x)); cellmask[idx] = cm.cpu().numpy()
        for c, fn in conds.items():
            xc = fn(x); F = dense_features(model, xc).double()           # (B,C,G,G)
            if c == 'orig':
                if feats is None: feats = np.zeros((N, G, G, F.shape[1]), np.float16)
                feats[idx] = F.permute(0, 2, 3, 1).half().cpu().numpy()
            for n in VARIANTS:
                for b in range(len(idx)):
                    pr = probes[(n, k[b])]
                    s = torch.einsum('chw,c->hw', F[b] - pr['mu'].view(-1, 1, 1), pr['w'])
                    maps[(n, c)][idx[b]] = s.float().cpu().numpy()
            pool = man['size'] // G
            maps[('intensity', c)][idx] = Fnn.avg_pool2d(xc, pool)[:, 0].cpu().numpy()
            filled = xc
            dog = (Fnn.avg_pool2d(filled, 5, 1, 2) - Fnn.avg_pool2d(filled, 63, 1, 31))
            maps[('dog_prior', c)][idx] = Fnn.avg_pool2d(dog, pool)[:, 0].cpu().numpy()
        print(f'pred {idx[-1]+1}/{N} {time.monotonic()-t0:.0f}s', flush=True)
    # PatchCore-style label-free dense baseline on original images (cosine kNN to training-fold cells)
    rng = np.random.default_rng(a.seed)
    knn = np.zeros((N, G, G), np.float16)
    for k in range(K):
        tr = np.where(fidx != k)[0]; te = np.where(fidx == k)[0]
        bank_all = feats[tr][cellmask[tr]]
        bank = torch.from_numpy(bank_all[rng.choice(len(bank_all), min(a.knn_bank, len(bank_all)), replace=False)]).to(dev).float()
        bank = Fnn.normalize(bank, dim=1)
        for s0 in range(0, len(te), 32):
            ii = te[s0:s0 + 32]
            q = Fnn.normalize(torch.from_numpy(feats[ii].reshape(-1, feats.shape[-1])).to(dev).float(), dim=1)
            d = 1 - (q @ bank.T).topk(3, dim=1).values.mean(1)
            knn[ii] = d.view(len(ii), G, G).cpu().numpy()
        print('knn fold', folds[k], flush=True)
    maps[('patch_knn', 'orig')] = knn
    rows = []
    for (n, c), m in maps.items():
        for i in range(N):
            cm = cellmask[i]; v = m[i][cm].astype(np.float64)
            rows.append(dict(uid=frame.uid[i], fold=frame.fold[i], method=n, cond=c,
                             burden_mean=float(v.mean()), burden_pos=float(np.maximum(v, 0).mean()),
                             burden_top10=float(np.sort(v)[-max(1, len(v) // 10):].mean())))
    pred = pd.DataFrame(rows)
    pp = run / 'predictions_frozen.csv'; pred.to_csv(pp, index=False)
    mp = run / 'dense_maps_frozen.npz'
    np.savez_compressed(mp, uids=frame.uid.to_numpy(str), cellmask=cellmask,
                        **{f'{n}__{c}': m for (n, c), m in maps.items()})
    (run / 'probe_diagnostics.json').write_text(json.dumps(
        {f'{n}|{folds[k]}': dict(lam=v['lam'], cv=v['cv'], train_synth_r2=v['train_synth_r2'])
         for (n, k), v in probes.items()}, indent=2))
    torch.save({f'{n}|{folds[k]}': dict(w=v['w'].cpu(), mu=v['mu'].cpu()) for (n, k), v in probes.items()}, run / 'probes.pt')
    (run / 'frozen_manifest.json').write_text(json.dumps(dict(
        predictions_sha256=hashlib.sha256(pp.read_bytes()).hexdigest(),
        maps_sha256=hashlib.sha256(mp.read_bytes()).hexdigest(), main_method='cdp_phys', main_burden='burden_mean',
        variants={k: [list(v[0]), v[1]] for k, v in VARIANTS.items()}, lams=LAMS, masks_read=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), elapsed_s=time.monotonic() - t0), indent=2))
    print('frozen', flush=True)


if __name__ == '__main__':
    main()
