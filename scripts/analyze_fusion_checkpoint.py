"""Measure actual checkpoint errors, PR curves and optional gate regions on FA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from scripts.run_paper_reproduction import atomic_json
from scripts.train_unet_multilabel import setup_logger
from scripts.fusion_evaluation_components import evaluation_components
from bs.fusion_diagnostics import GateRegionStatistics, ProbabilityHistogram, aggregate_errors, binary_error_record
from bs.multilabel import masks_to_paper_targets
from bs.recent_unet_methods import DifferenceGatedFusion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--fold", default="f1")
    parser.add_argument("--output", required=True, help="Output directory under project outputs")
    parser.add_argument("--save-probabilities", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    run, output = Path(args.run).resolve(), Path(args.output).resolve()
    output.relative_to(PROJECT_ROOT / "outputs")
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((run / "resolved_config.json").read_text())
    build_loader, build_model, _, _ = evaluation_components(config)
    if config["train"].get("max_val_samples") or not config["data"].get("exclude_val_augmented", True):
        raise ValueError("Diagnostics require complete clean-fold validation")
    config["runtime"]["num_workers"] = 2
    logger = setup_logger(output / "analysis.log", "fusion_checkpoint_analysis")
    loader = build_loader(config, args.fold, "val", logger)
    expected_samples = len(loader.dataset)
    if args.fold == "f1" and expected_samples != 444:
        raise RuntimeError(f"Expected 444 clean f1 samples, found {expected_samples}")
    checkpoint_path = run / args.fold / "checkpoints/best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    best_metrics = checkpoint["best_metrics"]
    shared_threshold = float(best_metrics["paper_sweep_best_threshold"])
    best_epoch = int(checkpoint["epoch"])
    model = build_model(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    device = torch.device(config["runtime"]["device"])
    model = model.to(device).eval()
    fusions = [module for module in model.modules() if isinstance(module, DifferenceGatedFusion)]
    for module in fusions:
        module.capture_diagnostics = True
    gate_statistics, histogram = GateRegionStatistics(), ProbabilityHistogram()
    per_image = []
    probability_dir = output / "probabilities"
    if args.save_probabilities:
        probability_dir.mkdir(exist_ok=True)
    started = time.monotonic()
    temp_rows = output / "per_image.jsonl.tmp"
    with torch.inference_mode(), temp_rows.open("w") as stream:
        for batch_index, batch in enumerate(loader):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=bool(config["train"]["amp"]) and device.type == "cuda"):
                logits = model(images)
            if isinstance(logits, tuple):
                logits = logits[0]
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite logits in checkpoint diagnostics")
            probabilities = logits.sigmoid()
            targets, valid = masks_to_paper_targets(masks, config["data"]["ignore_index"])
            targets_np, valid_np = targets.cpu().numpy().astype(bool), valid[:, 0].cpu().numpy().astype(bool)
            probabilities_np = probabilities.float().cpu().numpy()
            histogram.update(probabilities_np, targets_np, valid_np)
            thresholds = {"fixed_0.5": 0.5, "shared": shared_threshold}
            predictions = {name: probabilities >= threshold for name, threshold in thresholds.items()}
            for stage, module in enumerate(fusions):
                gate_statistics.update(stage, module.last_diagnostics, targets, valid, predictions["shared"])
                module.last_diagnostics = {}
            predictions_np = {name: prediction.cpu().numpy() for name, prediction in predictions.items()}
            for sample_index, sample_id in enumerate(batch["sample_id"]):
                sample = loader.dataset.samples[len(per_image)]
                record = {"sample_id": sample_id, "fold": args.fold, "image_path": str(sample.image_path),
                          "mask_path": str(sample.mask_path), "valid_pixels": int(valid_np[sample_index].sum()),
                          "protocols": {}}
                for name in thresholds:
                    record["protocols"][name] = {
                        f"lesion_{lesion + 1}": binary_error_record(predictions_np[name][sample_index, lesion],
                                                                    targets_np[sample_index, lesion], valid_np[sample_index])
                        for lesion in range(2)
                    }
                if args.save_probabilities:
                    path = probability_dir / f"{len(per_image):04d}.npz"
                    np.savez_compressed(path, probabilities=probabilities_np[sample_index].astype(np.float16),
                                        mask=masks[sample_index].cpu().numpy().astype(np.uint8))
                    record["probabilities_path"] = str(path)
                per_image.append(record)
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            if (batch_index + 1) % 10 == 0 or len(per_image) == expected_samples:
                logger.info("analyzed %d/%d validation images", len(per_image), expected_samples)
    temp_rows.replace(output / "per_image.jsonl")
    errors = {}
    for name in ("fixed_0.5", "shared"):
        errors[name] = {f"lesion_{lesion + 1}": aggregate_errors([row["protocols"][name][f"lesion_{lesion + 1}"] for row in per_image])
                        for lesion in range(2)}
        errors[name]["macro_global_dice"] = float(np.mean([errors[name][f"lesion_{lesion + 1}"]["global_pixel_dice"] for lesion in range(2)]))
    expected = {"fixed_0.5": float(best_metrics["paper_macro_dice"]),
                "shared": float(best_metrics["paper_sweep_best_macro_dice"])}
    differences = {name: abs(errors[name]["macro_global_dice"] - value) for name, value in expected.items()}
    matching = all(math.isclose(errors[name]["macro_global_dice"], value, rel_tol=1e-6, abs_tol=1e-6)
                   for name, value in expected.items())
    atomic_json(output / "probability_curves.json", histogram.compute())
    report = {"run": str(run), "fold": args.fold, "best_epoch": best_epoch, "validation_samples": len(per_image),
              "image_size": config["train"]["image_size"], "fixed_threshold": 0.5, "shared_threshold": shared_threshold,
              "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
              "sample_ids_sha256": hashlib.sha256("\n".join(row["sample_id"] for row in per_image).encode()).hexdigest(),
              "errors": errors, "gate_region_statistics": gate_statistics.compute(),
              "reported_macro": expected, "macro_absolute_differences": differences, "macro_matches_checkpoint": matching,
              "seconds": time.monotonic() - started,
              "boundary_definition": "GT-positive images only; exact 4-neighbor boundary pixels; NSD tolerance 2 pixels; empty predictions listed separately",
              "component_definition": "8-connected GT; detected when >=10% of component pixels predicted; bins use image-area fractions equivalent to 16/64/256 pixels at 256x256",
              "gate_definition": "channel-averaged local weight and feature statistics; ROI area occupancy preserved at each stage; not calibrated uncertainty",
              "gate_stage_order": "FPN: 0 shallow to 2 deep" if config["model"].get("backbone") else "U-Net: 0 deep to 4 shallow",
              "patient_grouping": "unavailable; sample IDs are not established patient IDs",
              "vessel_regions": "unavailable; no vessel ground truth inferred",
              "scope": "same clean validation used for checkpoint/threshold selection; diagnostic analysis, no independent test claim"}
    atomic_json(output / "summary.json", report)
    print(json.dumps({key: report[key] for key in ("validation_samples", "best_epoch", "macro_absolute_differences", "macro_matches_checkpoint", "seconds")}, indent=2))
    if not matching:
        raise RuntimeError("Diagnostic predictions do not match selected checkpoint macro; inspect output")


if __name__ == "__main__":
    main()
