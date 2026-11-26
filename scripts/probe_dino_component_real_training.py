"""Probe the full DINO512 component objective on its paired training batch."""

from __future__ import annotations

import argparse
import hashlib
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
from scripts import train_dinov3_multilabel as dino
from scripts import train_dino_fusion_paired as paired
from scripts.train_dino_component_supervised import build_model, build_loss
from scripts.run_paper_reproduction import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--optimizer-memory", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    paired.configure_training()
    root = PROJECT_ROOT / "outputs/overnight_unet_20261012/dino_component_transfer_512"
    plan = json.loads((root / "plan.json").read_text())
    if len(plan["jobs"]) != 1:
        raise ValueError("Probe requires one explicitly prepared candidate")
    config = yaml.safe_load((PROJECT_ROOT / plan["jobs"][0]["config"]).read_text())
    manifest = json.loads((root / "source_manifest.json").read_text())
    for name, expected in manifest["source_sha256"].items():
        if hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen source changed: {name}")
    device = torch.device(args.device)
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info()
        if free < 6800 * 1024**2:
            raise RuntimeError(f"Defer whole-model probe: only {free / 1024**2:.0f}MiB free; require6800MiB; keep existing trainers")
        torch.cuda.reset_peak_memory_stats()
    logger = logging.getLogger("dino_component_probe")
    train_loader = dino.build_loader(config, "f1", "train", logger)
    val_loader = dino.build_loader(config, "f1", "val", logger)
    seed_paired_epoch(config, 1, train_loader, val_loader)
    batch = next(iter(train_loader))
    if any(fold == "f1" for fold in batch["fold"]):
        raise RuntimeError("Validation image in training probe")
    model = initialize_paired_model(build_model, config)
    reference_path = PROJECT_ROOT / "outputs/overnight_unet_20261012/strong_baseline_512/runs/dino_baseline_vsubr_s42_f1_30_512/f1/initial_weight_fingerprints.json"
    reference = json.loads(reference_path.read_text())
    if initial_weight_fingerprints(model) != reference:
        raise RuntimeError("DINO512 original initialization differs")
    parameters = sum(p.numel() for p in model.parameters())
    model = model.to(device).train()
    criterion = build_loss(config).to(device)
    criterion.epoch, criterion.enabled = 11, True
    if not criterion.needs_image:
        raise RuntimeError("Original image-aware VS-UBR loss was lost")
    images, masks = batch["image"].to(device), batch["mask"].to(device)
    optimizer = dino.build_optimizer(model, config) if args.optimizer_memory else None
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    attempts = []
    started = time.time()
    for attempt in range(1, 7):
        model.zero_grad(set_to_none=True)
        criterion.reset_statistics()
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
            base_loss = criterion.base(logits, masks, image=images)
            total = criterion(logits, masks, image=images)
        auxiliary = total - base_loss
        if not bool(torch.isfinite(total)) or float(auxiliary.detach()) <= 0:
            raise FloatingPointError("Full component weight must be finite and active")
        scale = scaler.get_scale()
        scaler.scale(total).backward()
        if optimizer is not None:
            scaler.unscale_(optimizer)
        elif device.type == "cuda":
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(scale)
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        finite = all(bool(torch.isfinite(g).all()) for g in gradients)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["clip_grad_norm"]))
        finite = finite and bool(torch.isfinite(grad_norm))
        attempts.append({"attempt": attempt, "amp_scale": scale, "base_loss": float(base_loss.detach()),
            "total_loss": float(total.detach()), "auxiliary_loss": float(auxiliary.detach()),
            "all_active_parameter_gradients_finite": finite,
            "gradient_norm": float(grad_norm) if finite else None})
        if optimizer is not None:
            if finite:
                scaler.step(optimizer)
            scaler.update()
        elif device.type == "cuda":
            scaler.update(new_scale=scale if finite else scale / 2.)
        if finite:
            break
    if not finite:
        raise FloatingPointError("Gradient probe remained nonfinite after AMP backoffs")
    report = {"status": "passed", "device": str(device), "mode": criterion.mode,
        "initial_original_state_tensors_matched": len(reference), "parameter_count": parameters,
        "sample_ids": list(batch["sample_id"]), "folds": list(batch["fold"]), "images_shape": list(images.shape),
        "images_sha256": hashlib.sha256(batch["image"].contiguous().numpy().tobytes()).hexdigest(),
        "masks_sha256": hashlib.sha256(batch["mask"].contiguous().numpy().tobytes()).hexdigest(),
        "active_gradient_tensors": len(gradients), "attempts": attempts,
        "component_statistics": criterion.statistics(), "original_image_aware_loss_preserved": True,
        "optimizer_steps": int(optimizer is not None), "optimizer_states": len(optimizer.state) if optimizer else 0,
        "seconds": time.time() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2 if device.type == "cuda" else None,
        "scope": "Original paired epoch1 training batch8 at512; full CAT weight at objective epoch11; original full model and image-aware loss; temporary probe only, no accuracy training or validation score; one batch does not prove all-epoch peak memory"}
    suffix = "_optimizer" if args.optimizer_memory else ""
    atomic_json(root / f"real_batch_{device.type}{suffix}_probe.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
