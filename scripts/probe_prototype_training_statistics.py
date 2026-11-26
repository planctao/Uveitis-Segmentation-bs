"""Audit PDR scales and eligibility using the paired training data stream."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from bs.paired_experiments import initialize_paired_model, seed_paired_epoch
from bs.prototype_regularization import prototype_discriminative_terms
from scripts.train_prototype_regularized_multilabel import build_model
from scripts import train_unet_multilabel as training
from scripts.run_paper_reproduction import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--feature-batches", type=int, default=4)
    args = parser.parse_args()
    if not 0 < args.feature_batches <= args.batches:
        parser.error("Require 0 < feature-batches <= batches")
    torch.set_num_threads(2)
    root = PROJECT_ROOT / "outputs/overnight_unet_20261012/prototype_semantic_preparation"
    root.mkdir(parents=True, exist_ok=True)
    config_path = PROJECT_ROOT / "outputs/overnight_unet_20261012/prototype_screen/input_configs/sward_pdr_joint_unit_s42_f1_30_256.yaml"
    config = yaml.safe_load(config_path.read_text())
    logger = logging.getLogger("pdr_probe")
    train_loader = training.build_loader(config, "f1", "train", logger)
    val_loader = training.build_loader(config, "f1", "val", logger)
    seed_paired_epoch(config, 1, train_loader, val_loader)
    batches, eligibility, counts = [], Counter(), []
    for i, batch in enumerate(train_loader, 1):
        assert all(fold != "f1" for fold in batch["fold"])
        count = torch.bincount(batch["mask"].flatten(), minlength=4)[:4].tolist()
        counts.append({"batch": i, "sample_ids": list(batch["sample_id"]), "folds": list(batch["fold"]),
                       "palette_pixel_counts": count})
        for label, n in enumerate(count):
            eligibility[str(label)] += int(n >= 2)
        if i <= args.feature_batches:
            batches.append(batch)
        if i == args.batches:
            break
    checkpoint = PROJECT_ROOT / "outputs/edg_validation_20261011/runs/paired_concat_s42_f1_30_256/f1/checkpoints/best.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    results = []
    for snapshot in ["initial_seed42", f"trained_baseline_epoch{saved['epoch']}"]:
        model = initialize_paired_model(build_model, config).train()
        if snapshot.startswith("trained"):
            model.load_state_dict(saved["model"], strict=True)
        criterion = training.build_loss(config)
        for index, batch in enumerate(batches, 1):
            started = time.time()
            with torch.no_grad():
                model(batch["image"])
                features = model._cached_feature.detach()
                model._cached_feature = None
            features.requires_grad_(True)
            loss = criterion(model.decoder.seg_layers[-1](features), batch["mask"])
            base_gradient = torch.autograd.grad(loss, features, retain_graph=True)[0]
            rows = {}
            for normalize in [False, True]:
                terms = prototype_discriminative_terms(features, batch["mask"], normalize_features=normalize)
                auxiliary = terms["separation"] + terms["compactness"]
                gradient = torch.autograd.grad(auxiliary, features, retain_graph=True)[0]
                rows[str(normalize)] = {"separation": float(terms["separation"].detach()),
                    "compactness": float(terms["compactness"].detach()),
                    "auxiliary_to_segmentation_loss_ratio": float(auxiliary.detach() / loss.detach()),
                    "feature_gradient_norm": float(gradient.norm()),
                    "feature_gradient_norm_ratio_to_segmentation": float(gradient.norm() / base_gradient.norm().clamp_min(1e-12)),
                    "cosine_with_segmentation_feature_gradient": float(F.cosine_similarity(
                        gradient.flatten(), base_gradient.flatten(), dim=0, eps=1e-12)),
                    "finite_gradients": bool(torch.isfinite(gradient).all()),
                    "eligible_classes": terms["eligible_classes"].tolist()}
            results.append({"snapshot": snapshot, "batch": index, "base_loss": float(loss.detach()),
                            "base_feature_gradient_norm": float(base_gradient.norm()),
                            "results": rows, "seconds": time.time() - started})
            print(json.dumps(results[-1]), flush=True)
            del features, loss, base_gradient, terms, auxiliary, gradient
        del model
    report = {"scope": "CPU FP32 training-only paired epoch1 stream probe; no optimizer updates; no validation inference or tuning; decoder-local feature gradients do not measure full parameter gradients",
              "batches_inspected": len(counts), "samples_inspected": sum(len(row["sample_ids"]) for row in counts),
              "feature_batches": len(batches), "paired_data_seed": config["runtime"]["paired_data_seed"],
              "runtime_workers": config["runtime"]["num_workers"], "eligibility": dict(eligibility),
              "counts": counts, "feature_probes": results}
    atomic_json(root / "paired_training_stream_probe.json", report)


if __name__ == "__main__":
    main()
