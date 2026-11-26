"""Verify complete FA paper models at the existing 768px baseline resolution.

Runs one real overlap/lesion2 example through the complete training objective,
backward, optimizer step, and no-grad evaluation. Reports memory and finite
gradients, not Dice claims. Dataset files are read-only throughout this audit.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from scripts.run_paper_reproduction import atomic_json, process_alive, process_start
from scripts.train_unet_multilabel import build_loader, build_loss, build_model, build_optimizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-pid", type=int)
    parser.add_argument("--output", default="outputs/paper_reproduction_20261011/high_resolution_audit.json")
    parser.add_argument("--methods", nargs="+", default=None)
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    torch.set_num_threads(2)
    torch.hub.set_dir(str(PROJECT_ROOT / "outputs/paper_reproduction_20261011/weights/torch/hub"))
    if args.wait_for_pid:
        start = process_start(args.wait_for_pid)
        while process_alive(args.wait_for_pid, start):
            time.sleep(5)
    config = yaml.safe_load(Path("configs/paper_reproduction/nnunet_control_768.yaml").read_text())
    config["runtime"]["num_workers"] = 0
    config["train"]["batch_size"] = 1
    loader = build_loader(config, "f1", "val", logging.getLogger("paper_preflight"))
    batch = None
    for candidate in loader:
        if ((candidate["mask"] == 2) | (candidate["mask"] == 3)).any():
            batch = candidate
            break
    if batch is None:
        raise RuntimeError("No lesion2-positive real validation image found for preflight")
    images, masks = batch["image"].cuda(), batch["mask"].cuda()
    print("real image selected", images.shape, "labels", masks.unique().tolist(), flush=True)
    rows = []
    methods = args.methods or ["condseg_cr", "condseg", "pbeunet", "medcagd", "udmamba", "spectralmamba", "puunet", "ukan", "mkunet"]
    for method in methods:
        config = yaml.safe_load(Path(f"configs/paper_reproduction/{method}.yaml").read_text())
        config["train"].update(image_size=[768, 768], batch_size=1, grad_accum_steps=8)
        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        model = build_model(config).cuda().train()
        criterion = build_loss(config)
        optimizer = build_optimizer(model, config)
        with torch.autocast("cuda", enabled=config["train"]["amp"]):
            logits = model(images)
            loss = criterion(logits, masks)
            if hasattr(model, "training_loss"):
                loss = model.training_loss(loss, images, masks, criterion)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{method} has a nonfinite 768px training loss")
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError(f"{method} has nonfinite 768px gradients")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", enabled=config["train"]["amp"]):
            output = model(images)
        if output.shape != (1, 2, 768, 768) or not torch.isfinite(output).all():
            raise FloatingPointError(f"{method} has invalid 768px no-grad evaluation output")
        row = {"method": method, "resolution": 768, "batch_size": 1, "pretrained": config["model"].get("pretrained", False),
               "loss": float(loss.detach()), "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
               "finite_gradients": True, "optimizer_step_verified": True, "finite_no_grad_output": True,
               "seconds": time.monotonic() - started}
        rows.append(row)
        atomic_json(Path(args.output), rows)
        print(json.dumps(row), flush=True)
        del model, optimizer, gradients, logits, loss, output
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
