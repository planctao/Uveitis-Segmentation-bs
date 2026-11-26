"""seg26 evaluation: per-checkpoint (optionally flip-TTA) probability caches + ensemble metrics.

Usage:
  python scripts/seg26_eval.py --fold f1 --ckpt runs/A/f1/checkpoints/best.pt [--ckpt ...] --tta --cache-dir outputs/seg26_eval_cache
Each checkpoint's sigmoid probabilities on the clean validation fold are cached (float16, .npy) so
that ensembles can be recombined cheaply.  Reports independent per-lesion threshold sweep macro and
fixed-0.5 macro for every member and for the probability-average ensemble.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_dinov3_multilabel as T  # noqa: E402
from bs.multilabel import masks_to_paper_targets  # noqa: E402

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90]


def cache_name(ckpt: str, tta: bool) -> str:
    h = hashlib.md5(str(Path(ckpt).resolve()).encode()).hexdigest()[:10]
    parts = Path(ckpt).parts
    tag = "_".join(parts[-4:-2]) if len(parts) >= 4 else Path(ckpt).stem
    return f"{tag}_{Path(ckpt).stem}_{'tta' if tta else 'raw'}_{h}.npy"


@torch.no_grad()
def predict(ckpt: str, fold: str, tta: bool, device: torch.device, logger) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    config = state["config"]
    config["train"]["max_val_samples"] = None
    config["runtime"]["num_workers"] = 8
    model = T.build_model(config, load_pretrained=False)
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    loader = T.build_loader(config, fold, "val", logger)
    probs, targets, valids = [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        mask = batch["mask"]
        with torch.autocast(device_type="cuda", enabled=True):
            out = T.primary_logits(model(images)) if hasattr(T, "primary_logits") else model(images)
            if isinstance(out, (tuple, list)):
                out = out[0]
            logits = [out.float()]
            if tta:
                for dims in [(3,), (2,), (2, 3)]:
                    o = model(torch.flip(images, dims=dims))
                    if isinstance(o, (tuple, list)):
                        o = o[0]
                    logits.append(torch.flip(o.float(), dims=dims))
        prob = torch.stack([torch.sigmoid(l) for l in logits]).mean(0)
        probs.append(prob.half().cpu().numpy())
        t, v = masks_to_paper_targets(mask, 255)
        targets.append(t.bool().numpy())
        valids.append(v.expand_as(t).bool().numpy())
    return np.concatenate(probs), np.concatenate(targets), np.concatenate(valids)


def metrics(prob: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict:
    # chunked accumulation to keep GPU memory small (training may share the GPU)
    nthr = len(THRESHOLDS)
    inter = torch.zeros(nthr, 2, dtype=torch.float64, device="cuda")
    psum = torch.zeros(nthr, 2, dtype=torch.float64, device="cuda")
    tsum = torch.zeros(2, dtype=torch.float64, device="cuda")
    thr_t = torch.tensor(THRESHOLDS, device="cuda").view(-1, 1, 1, 1)
    for s in range(0, prob.shape[0], 8):
        p = torch.from_numpy(np.asarray(prob[s:s + 8])).cuda().float()
        v = torch.from_numpy(np.asarray(valid[s:s + 8])).cuda()
        t = torch.from_numpy(np.asarray(target[s:s + 8])).cuda() & v
        tsum += t.sum(dim=(0, 2, 3)).double()
        for i, thr in enumerate(THRESHOLDS):
            pred = (p >= thr) & v
            inter[i] += (pred & t).sum(dim=(0, 2, 3)).double()
            psum[i] += pred.sum(dim=(0, 2, 3)).double()
    dice = (2 * inter / (psum + tsum.view(1, 2)).clamp_min(1)).cpu().numpy()
    i1, i2 = int(dice[:, 0].argmax()), int(dice[:, 1].argmax())
    i5 = THRESHOLDS.index(0.50)
    return {
        "ind_macro": float((dice[i1, 0] + dice[i2, 1]) / 2), "ind_d1": float(dice[i1, 0]), "ind_d2": float(dice[i2, 1]),
        "thr1": THRESHOLDS[i1], "thr2": THRESHOLDS[i2],
        "fixed05_macro": float(dice[i5].mean()), "fixed05_d1": float(dice[i5, 0]), "fixed05_d2": float(dice[i5, 1]),
        "shared_macro": float(dice.mean(1).max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--cache-dir", default="outputs/seg26_eval_cache")
    ap.add_argument("--no-ensemble", action="store_true")
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("seg26_eval")
    device = torch.device("cuda")
    cache = ROOT / args.cache_dir / args.fold
    cache.mkdir(parents=True, exist_ok=True)
    target = valid = None
    members = {}
    for ck in args.ckpt:
        f = cache / cache_name(ck, args.tta)
        if f.exists() and (cache / "target.npy").exists():
            prob = np.load(f, mmap_mode="r")
            target, valid = np.load(cache / "target.npy"), np.load(cache / "valid.npy")
        else:
            prob, target, valid = predict(ck, args.fold, args.tta, device, logger)
            np.save(f, prob)
            if not (cache / "target.npy").exists():
                np.save(cache / "target.npy", target)
                np.save(cache / "valid.npy", valid)
        members[ck] = metrics(prob, target, valid)
        print(json.dumps({"member": ck, "tta": args.tta, **members[ck]}))
    result = {"fold": args.fold, "tta": args.tta, "members": members}
    if len(args.ckpt) > 1 and not args.no_ensemble:
        acc = None
        for ck in args.ckpt:
            p = np.load(cache / cache_name(ck, args.tta)).astype(np.float32)
            acc = p if acc is None else acc + p
        acc /= len(args.ckpt)
        result["ensemble"] = metrics(acc, target, valid)
        print(json.dumps({"ensemble": len(args.ckpt), "tta": args.tta, **result["ensemble"]}))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
