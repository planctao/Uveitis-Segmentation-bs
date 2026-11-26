"""Check the complete component-supervised model on a real paired train batch."""

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from bs.paired_experiments import initialize_paired_model, initial_weight_fingerprints, seed_paired_epoch
from scripts.train_component_supervised_multilabel import build_model
from scripts import train_unet_multilabel as training
from scripts.run_paper_reproduction import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--optimizer-memory", action="store_true",
                        help="Take one ephemeral optimizer step to include actual optimizer-state memory")
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = PROJECT_ROOT / "outputs/overnight_unet_20261012"
    config = yaml.safe_load((root / "component_screen/input_configs/catmil_lesion2_joint_s42_f1_30_256.yaml").read_text())
    logger = logging.getLogger("component_probe")
    train_loader = training.build_loader(config, "f1", "train", logger)
    val_loader = training.build_loader(config, "f1", "val", logger)
    seed_paired_epoch(config, 1, train_loader, val_loader)
    batch = next(iter(train_loader))
    assert all(fold != "f1" for fold in batch["fold"])
    model = initialize_paired_model(build_model, config)
    actual = initial_weight_fingerprints(model)
    reference = json.loads((PROJECT_ROOT / "outputs/edg_validation_20261011/runs/paired_concat_s42_f1_30_256/f1/initial_weight_fingerprints.json").read_text())
    assert actual == reference
    device = torch.device(args.device)
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info()
        if free < 2600 * 1024 * 1024:
            raise RuntimeError(f"Whole-model probe deferred: only{free / 1024**2:.0f}MiB free; keep current trainers")
        torch.cuda.reset_peak_memory_stats()
    model = model.to(device).train()
    optimizer = training.build_optimizer(model, config) if args.optimizer_memory else None
    # Exercise both auxiliaries at full CAT weight, without changing a real
    # run's first-epoch schedule or taking an optimizer step.
    model.epoch = 11
    images, masks = batch["image"].to(device), batch["mask"].to(device)
    criterion = training.build_loss(config).to(device)
    started = time.time()
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits = model(images)
        base_loss = criterion(logits, masks)
        total = model.training_loss(base_loss, images, masks, criterion)
    total.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    finite = bool(torch.isfinite(total) and all(torch.isfinite(g).all() for g in gradients))
    assert finite
    if optimizer is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["clip_grad_norm"]))
        optimizer.step()
    report = {"device": str(device), "precision": "unscaled AMP backward" if device.type == "cuda" else "FP32",
        "sample_ids": list(batch["sample_id"]), "folds": list(batch["fold"]),
        "images_shape": list(images.shape), "base_loss": float(base_loss.detach()),
        "total_loss": float(total.detach()), "all_active_parameter_gradients_finite": finite,
        "active_gradient_tensors": len(gradients), "initial_baseline_tensors_matched": len(actual),
        "component_statistics": model.component_statistics(), "seconds": time.time() - started,
        "optimizer_steps": int(optimizer is not None),
        "optimizer_states": len(optimizer.state) if optimizer is not None else 0,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "scope": ("paired epoch1 training batch; whole-model forward/backward at full CAT/MIL weight; "
                  + ("one ephemeral optimizer step allocating real optimizer state; " if optimizer is not None else "no optimizer step; ")
                  + "no validation inference or accuracy training; one batch does not prove all30 epochs' peak memory")}
    suffix = "_optimizer" if args.optimizer_memory else ""
    atomic_json(root / f"component_semantic_preparation/real_batch_{device.type}{suffix}_probe.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
