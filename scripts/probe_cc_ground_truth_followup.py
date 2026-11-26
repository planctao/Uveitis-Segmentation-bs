"""Probe both independent follow-up modes on matched real training batches."""

import argparse
import copy
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
from scripts import train_unet_multilabel as training
from scripts.train_cc_ground_truth_followup import build_model
from scripts.run_dino_component_transfer_gate import assert_frozen
from scripts.run_paper_reproduction import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--optimizer-memory", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = PROJECT_ROOT / "outputs/overnight_unet_20261012/cc_gt_followup"
    device = torch.device(args.device)
    prepared = (root / "plan.json").exists()
    if prepared:
        assert_frozen(root)
    elif device.type != "cpu":
        raise RuntimeError("CUDA admission requires the fully prepared frozen plan")
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info()
        if free < 3200 * 1024**2:
            raise RuntimeError(f"Defer whole-model probe: only {free / 1024**2:.0f} MiB free")
    results = {}
    for seed, mode in [(42, "global_only"), (2026, "gt_only")]:
        name = f"ccfollowup_lesion2_{mode}_s{seed}_f1_30_256"
        if prepared:
            config = yaml.safe_load((root / "input_configs" / f"{name}.yaml").read_text())
        else:
            # Ephemeral CPU operator preparation does not create an accuracy plan.
            baseline = PROJECT_ROOT / f"outputs/edg_validation_20261011/runs/paired_concat_s{seed}_f1_30_256"
            config = copy.deepcopy(json.loads((baseline / "resolved_config.json").read_text()))
            recipe_path = root.parent / "bicc_screen/runs/bicc_lesion2_gt_only_s42_f1_30_256/resolved_config.json"
            weight = json.loads(recipe_path.read_text())["model"]["bidirectional_component_supervision"]["weight"]
            config["model"]["name"] = "cc_ground_truth_followup_controlled"
            config["model"]["cc_ground_truth_followup"] = {"mode": mode, "weight": weight}
        logger = logging.getLogger("cc_followup_probe")
        train_loader = training.build_loader(config, "f1", "train", logger)
        val_loader = training.build_loader(config, "f1", "val", logger)
        seed_paired_epoch(config, 1, train_loader, val_loader)
        batch = next(iter(train_loader))
        assert all(fold != "f1" for fold in batch["fold"])
        model = initialize_paired_model(build_model, config)
        reference_path = PROJECT_ROOT / f"outputs/edg_validation_20261011/runs/paired_concat_s{seed}_f1_30_256/f1/initial_weight_fingerprints.json"
        assert initial_weight_fingerprints(model) == json.loads(reference_path.read_text())
        model = model.to(device).train()
        criterion = training.build_loss(config).to(device)
        optimizer = training.build_optimizer(model, config) if args.optimizer_memory else None
        images, masks = batch["image"].to(device), batch["mask"].to(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.time()
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
            base = criterion(logits, masks)
            loss = model.training_loss(base, images, masks, criterion)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert bool(torch.isfinite(loss)) and all(bool(torch.isfinite(g).all()) for g in gradients)
        if optimizer:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["clip_grad_norm"]))
            optimizer.step()
        results[mode] = {
            "seed": seed, "sample_ids": list(batch["sample_id"]), "folds": list(batch["fold"]),
            "image_sha256": hashlib.sha256(batch["image"].contiguous().numpy().tobytes()).hexdigest(),
            "mask_sha256": hashlib.sha256(batch["mask"].contiguous().numpy().tobytes()).hexdigest(),
            "image_shape": list(images.shape), "base_loss": float(base.detach()), "total_loss": float(loss.detach()),
            "original_initial_tensors_matched": 292, "all_active_gradients_finite": True,
            "active_gradient_tensors": len(gradients), "optimizer_states": len(optimizer.state) if optimizer else 0,
            "seconds": time.time() - started,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
            "statistics": model.instance_statistics()}
        del model, criterion, optimizer, images, masks, logits, base, loss, gradients, train_loader, val_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {"status": "passed", "device": str(device), "optimizer_memory": args.optimizer_memory,
              "prepared_accuracy_plan_exists": prepared,
              "results": results, "scope": "Two whole-model ephemeral training-batch optimizer probes; no accuracy training or validation inference"}
    if device.type == "cuda":
        cpu = json.loads((root / "real_batch_cpu_optimizer_probe.json").read_text())
        if cpu["status"] != "passed" or set(cpu["results"]) != set(results):
            raise RuntimeError("Require both successful CPU operator probes")
        fields = ["seed", "sample_ids", "folds", "image_sha256", "mask_sha256", "image_shape"]
        for mode, result in results.items():
            if any(result[key] != cpu["results"][mode][key] for key in fields):
                raise RuntimeError("CPU/CUDA training batches differ")
        report["CPU_training_batches_match"] = True
    atomic_json(root / f"real_batch_{device.type}_optimizer_probe.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
