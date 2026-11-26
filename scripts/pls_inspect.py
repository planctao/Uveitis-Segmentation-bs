"""Inspect learned PLS coefficients: does the photometric term actually contribute to the logit?"""
import logging
import sys
from pathlib import Path

import torch

ROOT = Path("/root/autodl-tmp/Uveitis-Segmentation-bs")
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import train_dinov3_multilabel as T  # noqa
from bs.multilabel import masks_to_paper_targets  # noqa

for name in ["best.pt", "best_ema.pt"]:
    state = torch.load(ROOT / "runs/seg26_pls_s6_f1/f1/checkpoints" / name, map_location="cpu", weights_only=False)
    cfg = state["config"]; cfg["runtime"]["num_workers"] = 4
    model = T.build_model(cfg, load_pretrained=False)
    model.load_state_dict(state["model"], strict=True)
    dev = torch.device("cuda"); model.to(dev).eval()
    w = model.decode_head.pls_head.coeff
    print(name, "bias", w.bias.data.view(2, -1).tolist(), "|W| per out", w.weight.data.flatten(1).norm(dim=1).tolist())
    loader = T.build_loader(cfg, "f1", "val", logging.getLogger("s"))
    dec = model.decode_head
    stats = {k: [] for k in ["a1", "a2", "c1", "c2", "photo_std1", "photo_std2", "base_std1", "base_std2", "flip1", "flip2", "bandflip1", "bandflip2"]}
    captured = {}

    def hook(mod, inp, out):
        captured["args"] = inp

    h = dec.pls_head.register_forward_hook(hook)
    n = 0
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(dev)
            with torch.autocast("cuda"):
                model(img)
            logits, feat, images, osz = captured["args"]
            ex = dec.pls_head.explain(logits, feat, images, osz)
            g, coef, base = ex["guides"], ex["coef"], ex["base"]
            photo = (coef * g.unsqueeze(1)).sum(2)
            full = base + photo
            t, v = masks_to_paper_targets(batch["mask"], 255)
            t = t.to(dev).bool()
            for c in range(2):
                stats[f"a{c+1}"].append(coef[:, c, 0].mean().item())
                stats[f"c{c+1}"].append(coef[:, c, 1].mean().item() if coef.shape[2] > 1 else 0)
                stats[f"photo_std{c+1}"].append(photo[:, c].std().item())
                stats[f"base_std{c+1}"].append(base[:, c].std().item())
                # fraction of pixels where the photometric term flips the decision (at logit 0)
                stats[f"flip{c+1}"].append(((base[:, c] > 0) != (full[:, c] > 0)).float().mean().item())
                near = (full[:, c].abs() < 2)
                stats[f"bandflip{c+1}"].append((((base[:, c] > 0) != (full[:, c] > 0)) & near).float().sum().item() / max(near.float().sum().item(), 1))
            n += img.shape[0]
            if n >= 96:
                break
    h.remove()
    print(name, {k: round(sum(v) / len(v), 4) for k, v in stats.items()})
