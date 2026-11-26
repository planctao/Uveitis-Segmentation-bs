"""Iso-contour hypothesis test for FA leakage annotations (CPU, cached predictions).

H0: the annotated leakage boundary is (close to) a super-level set of the smoothed fluorescence
intensity, i.e. GT ~= {I_s >= theta} inside a neighbourhood of the lesion.  If true, a full-resolution
image iso-contour should place boundaries better than the network's own probability iso-contour.

Stage 1 (--extract): dump grey images of the clean val fold in loader order -> gray.npy (uint8).
Stage 2: per image/channel with non-empty GT, inside support S_r = dilate(GT, r):
  model-oracle  : best per-image threshold on network probability
  image-oracle  : best per-image threshold on smoothed / background-normalised intensity
  model-global  : network at its global threshold (realistic)
Also a training-free 'iso-snap' (no GT): in the band of the model mask, re-decide pixels by the
image intensity level estimated on the model's own contour.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
CACHE = None
PROB = None
ARGS = None


def extract(cache: Path, ckpt: str, fold: str) -> None:
    import torch
    sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
    import train_dinov3_multilabel as T
    from bs.multilabel import masks_to_paper_targets
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = state["config"]; cfg["train"]["max_val_samples"] = None; cfg["runtime"]["num_workers"] = 16
    loader = T.build_loader(cfg, fold, "val", logging.getLogger("x"))
    tgt = np.load(cache / "target.npy", mmap_mode="r")
    grays, k = [], 0
    for batch in loader:
        img = batch["image"].float()
        g = img.mean(1)  # monotone in intensity for grey FA regardless of normalisation
        lo = g.flatten(1).quantile(0.001, dim=1).view(-1, 1, 1); hi = g.flatten(1).quantile(0.999, dim=1).view(-1, 1, 1)
        g = ((g - lo) / (hi - lo + 1e-6)).clamp(0, 1)
        t, _ = masks_to_paper_targets(batch["mask"], 255)
        assert np.array_equal(t.bool().numpy(), tgt[k:k + len(t)]), "loader order mismatch"
        k += len(t)
        grays.append((g * 255).round().byte().numpy())
    np.save(cache / "gray.npy", np.concatenate(grays))
    print("saved gray", k)


def dice_counts(pred, gt):
    return float((pred & gt).sum()), float(pred.sum()), float(gt.sum())


def best_level(feat, gt, support, n=48):
    vals = feat[support]
    if vals.size == 0:
        return (0.0, 0.0, float(gt.sum()))
    qs = np.unique(np.quantile(vals, np.linspace(0, 1, n)))
    best, bc = -1, None
    g = gt[support]
    for q in qs:
        p = vals >= q
        i, ps, ts = float((p & g).sum()), float(p.sum()), float(g.sum())
        d = 2 * i / max(ps + ts, 1)
        if d > best:
            best, bc = d, (i, ps, ts)
    return bc


def work(i):
    P = np.load(CACHE / PROB, mmap_mode="r"); T = np.load(CACHE / "target.npy", mmap_mode="r")
    V = np.load(CACHE / "valid.npy", mmap_mode="r"); G = np.load(CACHE / "gray.npy", mmap_mode="r")
    I = G[i].astype(np.float32) / 255.0
    feats = {f"s{s}": ndi.gaussian_filter(I, s) for s in (1, 2, 4)}
    bg = ndi.gaussian_filter(I, 32)
    feats["s2_bgsub"] = feats["s2"] - bg
    feats["s2_bgdiv"] = feats["s2"] / (bg + 0.05)
    res = {}
    for c in range(2):
        v = V[i, c]; gt = T[i, c] & v; p = P[i, c].astype(np.float32)
        thr = ARGS.thr[c]
        pred = (p >= thr) & v
        r = {}
        # ---------- training-free iso-snap (no GT used) ----------
        for band_r in (3, 6, 10):
            for mode in ("s2", "s2_bgsub"):
                f = feats[mode]
                if pred.any():
                    edge = pred & ~ndi.binary_erosion(pred, iterations=1)
                    # per connected component level = median intensity on its own contour
                    lab, k = ndi.label(pred)
                    lvl = np.full(k + 1, np.inf, np.float32)
                    if k:
                        med = ndi.median(f, labels=lab * edge, index=np.arange(1, k + 1))
                        lvl[1:] = np.asarray(med, np.float32)
                    # assign level to band pixels via nearest component
                    dil = ndi.binary_dilation(pred, iterations=band_r)
                    core = ndi.binary_erosion(pred, iterations=band_r)
                    _, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
                    near = lab[iy, ix]
                    band = dil & ~core
                    snap = core | (band & (f >= lvl[near]) & (p >= ARGS.low))
                    snap &= v
                else:
                    snap = pred
                r[f"snap_r{band_r}_{mode}"] = dice_counts(snap, gt)
        r["model_global"] = dice_counts(pred, gt)
        # ---------- oracle comparisons inside GT neighbourhood ----------
        if gt.any():
            for sr in (8, 24):
                S = ndi.binary_dilation(gt, iterations=sr) & v
                r[f"S{sr}_model_global"] = dice_counts(pred & S, gt)
                r[f"S{sr}_model_oracle"] = best_level(p, gt, S)
                for name, f in feats.items():
                    r[f"S{sr}_img_{name}"] = best_level(f, gt, S)
                # per-component image oracle (spatially varying level)
                lab, k = ndi.label(ndi.binary_dilation(gt, iterations=sr) & v)
                acc = np.zeros(3)
                for j in range(1, k + 1):
                    acc += np.array(best_level(feats["s2_bgsub"], gt, lab == j))
                r[f"S{sr}_img_s2_bgsub_percomp"] = tuple(acc)
                # combo: model-oracle level AND image level both free (product rule p*sigmoid)
        res[c] = r
    return i, res


def main():
    global CACHE, PROB, ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--prob", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--fold", default="f1")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--thr", type=float, nargs=2, default=[0.8, 0.7])
    ap.add_argument("--low", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--out", default=None)
    ARGS = ap.parse_args()
    CACHE = Path(ARGS.cache)
    if ARGS.extract:
        extract(CACHE, ARGS.ckpt, ARGS.fold); return
    PROB = ARGS.prob
    n = np.load(CACHE / "target.npy", mmap_mode="r").shape[0]
    agg = {0: {}, 1: {}}
    with Pool(ARGS.workers) as pool:
        for i, res in pool.imap_unordered(work, range(n), chunksize=2):
            for c in res:
                for k, v in res[c].items():
                    agg[c].setdefault(k, np.zeros(3)); agg[c][k] += np.array(v)
    out = {}
    for c in agg:
        out[f"lesion{c+1}"] = {k: round(float(2 * v[0] / max(v[1] + v[2], 1)), 4) for k, v in sorted(agg[c].items())}
    print(json.dumps(out, indent=1))
    if ARGS.out:
        Path(ARGS.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
