"""Error anatomy of a cached probability map (CPU): where is Dice lost?

1) global-threshold Dice vs per-image oracle threshold (headroom of adaptive thresholds)
2) FP / FN decomposition: boundary band (<=r px from GT edge) vs detached components
3) image-level: empty-GT images with FP, positive images missed entirely
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

cache = Path(sys.argv[1])
prob_file = sys.argv[2]
P = np.load(cache / prob_file, mmap_mode="r")
T = np.load(cache / "target.npy", mmap_mode="r")
V = np.load(cache / "valid.npy", mmap_mode="r")
THR = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]
R = 4  # boundary band radius in pixels (768 res)
out = {}
for c in range(2):
    n = P.shape[0]
    inter = np.zeros((n, len(THR))); ps = np.zeros((n, len(THR))); ts = np.zeros(n)
    for i in range(n):
        v = V[i, c]; t = T[i, c] & v; p = P[i, c].astype(np.float32)
        ts[i] = t.sum()
        for j, th in enumerate(THR):
            pr = (p >= th) & v
            inter[i, j] = (pr & t).sum(); ps[i, j] = pr.sum()
    dice_g = 2 * inter.sum(0) / (ps.sum(0) + ts.sum())
    jg = int(dice_g.argmax()); thr = THR[jg]
    # per-image oracle threshold (maximise image dice; empty GT -> minimise FP)
    best = []
    for i in range(n):
        if ts[i] == 0:
            best.append(int(ps[i].argmin()))
        else:
            best.append(int((2 * inter[i] / (ps[i] + ts[i])).argmax()))
    bi = np.array(best)
    I_o = inter[np.arange(n), bi].sum(); P_o = ps[np.arange(n), bi].sum()
    dice_oracle = 2 * I_o / (P_o + ts.sum())
    # error anatomy at the global threshold
    fp_band = fp_det = fn_band = fn_miss = tp = 0
    fp_img_empty = 0; fp_pix_empty = 0; missed_pos_img = 0; n_pos = int((ts > 0).sum())
    comp_fp = comp_tot = 0
    for i in range(n):
        v = V[i, c]; t = T[i, c] & v; pr = (P[i, c] >= thr) & v
        tp += (pr & t).sum()
        if t.any():
            band = ndi.binary_dilation(t, iterations=R) & ~ndi.binary_erosion(t, iterations=R)
        else:
            band = np.zeros_like(t)
        fp = pr & ~t; fn = t & ~pr
        lab, k = ndi.label(pr)
        if k:
            hit = np.zeros(k + 1, bool); hit[np.unique(lab[t])] = True; hit[0] = False
            det = (lab > 0) & ~hit[lab]
            comp_fp += int((~hit[1:]).sum()); comp_tot += k
        else:
            det = np.zeros_like(pr)
        fp_det += (fp & det).sum(); fp_band += (fp & ~det & band).sum()
        labt, kt = ndi.label(t)
        if kt:
            found = np.zeros(kt + 1, bool); found[np.unique(labt[pr])] = True; found[0] = False
            miss = (labt > 0) & ~found[labt]
        else:
            miss = np.zeros_like(t)
        fn_miss += (fn & miss).sum(); fn_band += (fn & ~miss & band).sum()
        if not t.any() and pr.any():
            fp_img_empty += 1; fp_pix_empty += pr.sum()
        if t.any() and not (pr & t).any():
            missed_pos_img += 1
    FP = ps[:, jg].sum() - inter[:, jg].sum(); FN = ts.sum() - inter[:, jg].sum()
    out[f"lesion{c+1}"] = {
        "global_thr": thr, "dice_global": round(float(dice_g[jg]), 4), "dice_oracle_per_image_thr": round(float(dice_oracle), 4),
        "oracle_thr_hist": {str(THR[j]): int((bi == j).sum()) for j in range(len(THR))},
        "TP": int(tp), "FP": int(FP), "FN": int(FN),
        "FP_detached_components_frac": round(float(fp_det / FP), 3), "FP_boundary_band_frac": round(float(fp_band / FP), 3),
        "FN_missed_components_frac": round(float(fn_miss / FN), 3), "FN_boundary_band_frac": round(float(fn_band / FN), 3),
        "pred_components_total": comp_tot, "pred_components_false": comp_fp,
        "pos_images": n_pos, "empty_images_with_FP": fp_img_empty, "FP_pixels_on_empty_images": int(fp_pix_empty),
        "pos_images_fully_missed": missed_pos_img,
        "dice_if_no_detached_FP": round(float(2 * tp / (2 * tp + FP - fp_det + FN)), 4),
        "dice_if_no_missed_comp": round(float(2 * (tp + fn_miss) / (2 * (tp + fn_miss) + FP + FN - fn_miss)), 4),
        "dice_if_perfect_band": round(float(2 * (tp + fn_band) / (2 * (tp + fn_band) + FP - fp_band + FN - fn_band)), 4),
    }
    print(json.dumps({f"lesion{c+1}": out[f"lesion{c+1}"]}), flush=True)
Path(sys.argv[3]).write_text(json.dumps(out, indent=2))
