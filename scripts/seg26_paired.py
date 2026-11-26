"""Paired per-image comparison of two cached prob files (same f1 val order).
Usage: python seg26_paired.py cache_dir probA.npy thrA1 thrA2 probB.npy thrB1 thrB2 [out.json]
Reports per-image Dice deltas, image-bootstrap CI of pooled-Dice delta, and boundary-band error split."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

cache = Path(sys.argv[1])
A, ta = np.load(cache / sys.argv[2], mmap_mode="r"), (float(sys.argv[3]), float(sys.argv[4]))
B, tb = np.load(cache / sys.argv[5], mmap_mode="r"), (float(sys.argv[6]), float(sys.argv[7]))
T = np.load(cache / "target.npy", mmap_mode="r"); V = np.load(cache / "valid.npy", mmap_mode="r")
N = T.shape[0]
# per-image stats: inter, psum, tsum for A and B, plus band FN/FP counts
st = np.zeros((N, 2, 2, 3))  # img, model, class, (inter, psum, tsum)
band = np.zeros((2, 2, 4))  # model, class, (fn_band, fn_all, fp_band, fp_all)
for i in range(N):
    for c in range(2):
        v = V[i, c]; t = T[i, c] & v
        edge = ndi.binary_dilation(t, iterations=4) & ~ndi.binary_erosion(t, iterations=4) if t.any() else np.zeros_like(t)
        for m, (P, th) in enumerate([(A, ta), (B, tb)]):
            p = (P[i, c].astype(np.float32) >= th[c]) & v
            st[i, m, c] = [(p & t).sum(), p.sum(), t.sum()]
            fn, fp = t & ~p, p & ~t
            band[m, c] += [(fn & edge).sum(), fn.sum(), (fp & edge).sum(), fp.sum()]


def pooled(s):
    return 2 * s[..., 0].sum(0) / np.maximum(s[..., 1].sum(0) + s[..., 2].sum(0), 1)


out = {}
rng = np.random.default_rng(0)
for c in range(2):
    dA, dB = pooled(st[:, 0, c]), pooled(st[:, 1, c])
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, N, N)
        boots.append(pooled(st[idx, 1, c]) - pooled(st[idx, 0, c]))
    boots = np.array(boots)
    pos = st[:, 0, c, 2] > 0
    di = lambda s: 2 * s[:, 0] / np.maximum(s[:, 1] + s[:, 2], 1)
    per = di(st[pos, 1, c]) - di(st[pos, 0, c])
    missA = int(((st[pos, 0, c, 0]) == 0).sum()); missB = int(((st[pos, 1, c, 0]) == 0).sum())
    out[f"lesion{c+1}"] = {
        "dice_A": float(dA), "dice_B": float(dB), "delta": float(dB - dA),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "p_boot_le0": float((boots <= 0).mean()),
        "n_pos": int(pos.sum()), "img_better": int((per > 0.005).sum()), "img_worse": int((per < -0.005).sum()),
        "missed_A": missA, "missed_B": missB,
        "fn_band_frac": [float(band[m, c, 0] / max(band[m, c, 1], 1)) for m in range(2)],
        "fn_pixels": [int(band[m, c, 1]) for m in range(2)], "fp_pixels": [int(band[m, c, 3]) for m in range(2)],
        "fn_band_pixels": [int(band[m, c, 0]) for m in range(2)], "fp_band_pixels": [int(band[m, c, 2]) for m in range(2)],
    }
b = np.array([[out[f"lesion{c+1}"]["dice_B"] - out[f"lesion{c+1}"]["dice_A"] for c in range(2)]])
out["macro_delta"] = float(b.mean())
print(json.dumps(out, indent=1))
if len(sys.argv) > 8:
    Path(sys.argv[8]).write_text(json.dumps(out, indent=1))
