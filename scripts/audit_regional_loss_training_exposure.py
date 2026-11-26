"""Replay one fixed training epoch's targets, with no model or inference."""

import csv
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
from scipy import ndimage
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from scripts import train_unet_multilabel as training
from bs.paired_experiments import seed_paired_epoch
from scripts.run_paper_reproduction import atomic_json


def main():
    torch.set_num_threads(2)
    root = PROJECT_ROOT / "outputs/overnight_unet_20261012/bicc_screen"
    run = root / "runs/bicc_lesion2_gt_only_s42_f1_30_256"
    output = root / "gt_region_epoch1_training_exposure.json"
    if output.exists():
        raise RuntimeError("Do not repeat an existing exposure audit")
    config = json.loads((run / "resolved_config.json").read_text())
    logger = logging.getLogger("regional_exposure")
    train_loader = training.build_loader(config, "f1", "train", logger)
    val_loader = training.build_loader(config, "f1", "val", logger)
    seed_paired_epoch(config, 1, train_loader, val_loader)
    categories = {"zero_components": 0, "one_component": 0, "multiple_components": 0}
    counts, totals, sample_ids = [], [0, 0], []
    largest_component_fractions = []
    target_digest = hashlib.sha256()
    started = time.time()
    batches = 0
    for batch in train_loader:
        assert all(fold != "f1" for fold in batch["fold"])
        masks = batch["mask"].numpy()
        target_digest.update(masks.tobytes())
        sample_ids.extend(batch["sample_id"])
        totals[0] += int(np.count_nonzero((masks == 1) | (masks == 3)))
        totals[1] += int(np.count_nonzero((masks == 2) | (masks == 3)))
        for mask in masks:
            labels, number = ndimage.label((mask == 2) | (mask == 3), structure=np.ones((3, 3)))
            counts.append(number)
            category = "zero_components" if number == 0 else "one_component" if number == 1 else "multiple_components"
            categories[category] += 1
            if number > 1:
                areas = np.bincount(labels.ravel())[1:]
                largest_component_fractions.append(float(areas.max() / areas.sum()))
        batches += 1
    with (run / "f1/metrics.csv").open() as stream:
        reference = next(csv.DictReader(stream))
    expected = [int(float(reference[f"train_paper_target_pixels_{i}"])) for i in [1, 2]]
    if totals != expected:
        raise RuntimeError(f"Replayed training target totals differ: {totals} vs {expected}")
    report = {
        "run": str(run), "epoch": 1, "seed": 42, "batches": batches,
        "sampled_images": len(counts), "unique_sample_ids": len(set(sample_ids)),
        "component_counts": categories, "total_GT_components": int(sum(counts)),
        "maximum_components_per_image": max(counts),
        "fraction_images_with_distinct_region_objective": categories["multiple_components"] / len(counts),
        "multi_component_median_largest_GT_area_fraction": float(np.median(largest_component_fractions)) if largest_component_fractions else None,
        "training_target_pixels": totals, "original_metric_target_pixels": expected,
        "original_epoch_target_totals_match": True, "mask_sequence_sha256": target_digest.hexdigest(),
        "seconds": time.time() - started,
        "scope": "Fixed original epoch1 sampler/augmentation target replay only. Regional and whole-image DiceCE are mathematically equal for zero or one GT component; exposure is sampled-image frequency, not gradient contribution or all-epoch causal evidence. No model, accuracy training, validation inference, or parameter/threshold choice."}
    atomic_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
