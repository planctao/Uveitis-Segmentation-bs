"""Check both NoiseUNet location arms on the original paired train batch."""

import argparse
import gc
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
from scripts.run_paper_reproduction import atomic_json
from scripts.train_noise_skip_multilabel import build_model
from scripts import train_unet_multilabel as training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--optimizer-memory", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = PROJECT_ROOT / "outputs/overnight_unet_20261012/noise_skip_screen"
    for name, expected in json.loads((root / "source_manifest.json").read_text())["source_sha256"].items():
        if hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen source changed: {name}")
    configs = [yaml.safe_load((PROJECT_ROOT / job["config"]).read_text())
               for job in json.loads((root / "plan.json").read_text())["jobs"]]
    device = torch.device(args.device)
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info()
        if free < 2600 * 1024**2:
            raise RuntimeError(f"Defer whole-model probe: only{free / 1024**2:.0f}MiB free; existing trainers retained")
    logger = logging.getLogger("noise_skip_real_probe")
    loader = training.build_loader(configs[0], "f1", "train", logger)
    val_loader = training.build_loader(configs[0], "f1", "val", logger)
    seed_paired_epoch(configs[0], 1, loader, val_loader)
    batch = next(iter(loader))
    if any(fold == "f1" for fold in batch["fold"]):
        raise RuntimeError("Held-out data in train probe")
    reference = json.loads((PROJECT_ROOT / "outputs/edg_validation_20261011/runs/paired_concat_s42_f1_30_256/f1/initial_weight_fingerprints.json").read_text())
    results = {}
    for config in configs:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        model = initialize_paired_model(build_model, config)
        if initial_weight_fingerprints(model) != reference:
            raise RuntimeError("Original same-seed tensors differ")
        model = model.to(device).train()
        model.set_epoch(1)
        optimizer = training.build_optimizer(model, config) if args.optimizer_memory else None
        criterion = training.build_loss(config).to(device)
        images, masks = batch["image"].to(device), batch["mask"].to(device)
        started = time.time()
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            output = model(images)
            loss = criterion(output, masks)
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        finite = bool(torch.isfinite(loss)) and all(bool(torch.isfinite(g).all()) for g in gradients)
        if not finite or model.injection_count != 5:
            raise FloatingPointError("Full-model gradients or five-stage noise injection invalid")
        if optimizer is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["clip_grad_norm"]))
            optimizer.step()
        results[model.mode] = {"initial_original_tensors_matched": len(reference),
            "loss": float(loss.detach()), "active_gradient_tensors": len(gradients),
            "all_active_gradients_finite": finite, "five_noise_injections": model.injection_count,
            "optimizer_steps": int(optimizer is not None), "optimizer_states": len(optimizer.state) if optimizer else 0,
            "seconds": time.time() - started,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2 if device.type == "cuda" else None}
        del model, optimizer, criterion, images, masks, output, loss, gradients
        gc.collect()
    report = {"status": "passed", "device": str(device), "sample_ids": list(batch["sample_id"]),
        "folds": list(batch["fold"]), "images_shape": list(batch["image"].shape),
        "images_sha256": hashlib.sha256(batch["image"].contiguous().numpy().tobytes()).hexdigest(),
        "masks_sha256": hashlib.sha256(batch["mask"].contiguous().numpy().tobytes()).hexdigest(), "results": results,
        "scope": "Two original whole-model modes on paired epoch1 train batch8; unscaled AMP onCUDA; optional ephemeral optimizer steps; no validation or accuracy training; one batch does not certify full-run peak memory"}
    suffix = "_optimizer" if args.optimizer_memory else ""
    atomic_json(root / f"real_batch_{device.type}{suffix}_probe.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
