"""Probe complete BiCC FA model on a paired real training batch."""

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
from scripts.train_bidirectional_component_multilabel import build_model
from scripts import train_unet_multilabel as training
from scripts.run_paper_reproduction import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--optimizer-memory", action="store_true")
    parser.add_argument("--gradient-ratios", action="store_true")
    parser.add_argument("--mode", choices=["bidirectional", "gt_only", "smoothed_pred"], default="bidirectional")
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = PROJECT_ROOT / "outputs/overnight_unet_20261012"
    config = yaml.safe_load((root / f"bicc_screen/input_configs/bicc_lesion2_{args.mode}_s42_f1_30_256.yaml").read_text())
    device = torch.device(args.device)
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info()
        if free < 2600 * 1024**2:
            raise RuntimeError(f"Defer whole-model probe: only {free / 1024**2:.0f} MiB free")
        torch.cuda.reset_peak_memory_stats()
    logger = logging.getLogger("bicc_real_batch_probe")
    train_loader = training.build_loader(config, "f1", "train", logger)
    val_loader = training.build_loader(config, "f1", "val", logger)
    seed_paired_epoch(config, 1, train_loader, val_loader)
    batch = next(iter(train_loader))
    assert all(fold != "f1" for fold in batch["fold"])
    model = initialize_paired_model(build_model, config)
    reference = json.loads((PROJECT_ROOT / "outputs/edg_validation_20261011/runs/paired_concat_s42_f1_30_256/f1/initial_weight_fingerprints.json").read_text())
    fingerprints = initial_weight_fingerprints(model)
    assert fingerprints == reference
    model = model.to(device).train()
    criterion = training.build_loss(config).to(device)
    optimizer = training.build_optimizer(model, config) if args.optimizer_memory else None
    images, masks = batch["image"].to(device), batch["mask"].to(device)
    started = time.time()
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits = model(images)
        base_loss = criterion(logits, masks)
        total_loss = model.training_loss(base_loss, images, masks, criterion)
    gradient_ratios = None
    if args.gradient_ratios:
        parameters = [p for p in model.parameters() if p.requires_grad]
        main = torch.autograd.grad(base_loss, parameters, retain_graph=True, allow_unused=True)
        auxiliary = torch.autograd.grad(total_loss - base_loss, parameters, retain_graph=True, allow_unused=True)
        main_sq = sum(float(g.square().sum()) for g in main if g is not None)
        aux_sq = sum(float(g.square().sum()) for g in auxiliary if g is not None)
        dot = sum(float(a.mul(b).sum()) for a, b in zip(main, auxiliary, strict=True) if a is not None and b is not None)
        gradient_ratios = {"auxiliary_to_main_parameter_gradient_norm": (aux_sq / max(main_sq, 1e-20))**.5,
                           "cosine": dot / max((main_sq * aux_sq)**.5, 1e-20),
                           "all_active_main_auxiliary_gradients_finite": all(torch.isfinite(g).all() for gs in [main, auxiliary] for g in gs if g is not None)}
        del parameters, main, auxiliary
    total_loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    finite = bool(torch.isfinite(total_loss) and all(torch.isfinite(g).all() for g in gradients))
    assert finite
    if gradient_ratios:
        assert gradient_ratios["all_active_main_auxiliary_gradients_finite"]
    if optimizer is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["clip_grad_norm"]))
        optimizer.step()
    report = {"device": str(device), "precision": "unscaled CUDA AMP" if device.type == "cuda" else "FP32",
        "sample_ids": list(batch["sample_id"]), "folds": list(batch["fold"]), "images_shape": list(images.shape),
        "original_initial_tensors_match": True, "original_initial_tensor_count": len(fingerprints),
        "base_loss": float(base_loss.detach()), "total_loss": float(total_loss.detach()),
        "active_gradient_tensors": len(gradients), "all_active_gradients_finite": finite,
        "gradient_ratios": gradient_ratios, "instance_statistics": model.instance_statistics(),
        "optimizer_steps": int(optimizer is not None), "optimizer_states": len(optimizer.state) if optimizer is not None else 0,
        "seconds": time.time() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
        "scope": "Whole model on paired epoch1 training batch; optional single ephemeral optimizer step; no validation inference, accuracy training or parameter tuning; one batch does not certify all30epoch memory"}
    settings = config["model"]["bidirectional_component_supervision"]
    suffix = f"_{args.mode}_weight{settings['weight']:g}"
    suffix += "_optimizer" if args.optimizer_memory else ""
    suffix += "_gradients" if args.gradient_ratios else ""
    atomic_json(root / f"bicc_semantic_preparation/real_batch_{device.type}{suffix}_probe.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
