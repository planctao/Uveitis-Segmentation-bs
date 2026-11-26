"""Read-only audit of which annotated components survive mask subsampling.

Components are raster connected regions, not established clinical instances.
This measures preprocessing information loss, not a model's recall.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from scripts.run_paper_reproduction import atomic_json
from scripts.train_unet_multilabel import build_loader, setup_logger


def component_sampling(target: np.ndarray, size: tuple[int, int]) -> tuple[dict, np.ndarray]:
    labels, count = ndimage.label(target, structure=np.ones((3, 3)))
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    # Sample component IDs on the exact nearest-neighbor mask grid. Counting
    # overlap after expanding the low-resolution mask would let another region
    # incorrectly stand in for a component that was never sampled.
    sampled = F.interpolate(torch.from_numpy(labels).float()[None, None], size=size, mode="nearest")[0, 0].numpy().astype(np.int64)
    selected = np.bincount(sampled.ravel(), minlength=count + 1)
    coarse_labels, coarse_count = ndimage.label(sampled > 0, structure=np.ones((3, 3)))
    pairs = np.unique(np.c_[sampled[sampled > 0], coarse_labels[sampled > 0]], axis=0)
    fragments = np.bincount(pairs[:, 0], minlength=count + 1)
    merged = np.bincount(pairs[:, 1], minlength=coarse_count + 1)
    fractions = areas[1:] / target.size
    bins = np.searchsorted(np.array([16, 64, 256]) / 65536, fractions, side="left")
    by_size = {}
    for index, name in enumerate(("tiny", "small", "medium", "large")):
        subset = bins == index
        by_size[name] = {"source_components": int(subset.sum()),
                         "sampled_components": int((subset & (selected[1:] > 0)).sum()),
                         "lost_components": int((subset & (selected[1:] == 0)).sum())}
    return {"source_pixels": int(target.sum()), "resized_pixels": int((sampled > 0).sum()),
            "source_components": count, "resized_components": coarse_count,
            "source_components_split_after_resize": int((fragments[1:] > 1).sum()),
            "resized_components_merging_source_regions": int((merged[1:] > 1).sum()),
            "source_components_by_size": by_size}, sampled > 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="outputs/edg_validation_20261011/runs/paired_concat_s42_f1_30_256")
    parser.add_argument("--output", default="outputs/edg_validation_20261011/target_component_resizing_audit.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    run, output = (PROJECT_ROOT / args.run).resolve(), (PROJECT_ROOT / args.output).resolve()
    output.relative_to(PROJECT_ROOT / "outputs")
    output.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads((run / "resolved_config.json").read_text())
    if config["data"].get("label_values") != [0, 1, 2, 3]:
        raise ValueError("Component audit requires the existing two-lesion label mapping")
    if not config["data"].get("exclude_val_augmented") or config["train"].get("max_val_samples"):
        raise ValueError("Component audit requires full clean validation")
    logger = setup_logger(output.with_suffix(".log"), "target_component_resizing")
    dataset = build_loader(config, "f1", "val", logger).dataset
    if len(dataset) != 444:
        raise RuntimeError("Expected all 444 clean f1 images")
    records = []
    for sample in dataset.samples:
        mask = dataset._read_mask(sample.mask_path)
        _, resized = dataset._resize(torch.zeros(3, *dataset.image_size), mask)
        row = {"sample_id": sample.sample_id, "source_shape": list(mask.shape), "lesions": {}}
        for lesion in (1, 2):
            target = ((mask == lesion) | (mask == 3)).numpy()
            measured, sampled = component_sampling(target, dataset.image_size)
            expected = ((resized == lesion) | (resized == 3)).numpy()
            if not np.array_equal(sampled, expected):
                raise RuntimeError(f"Component sampling grid differs from dataset for {sample.sample_id}")
            row["lesions"][f"lesion_{lesion}"] = measured
        records.append(row)
    totals = {}
    for lesion in ("lesion_1", "lesion_2"):
        entries = [row["lesions"][lesion] for row in records]
        totals[lesion] = {key: sum(entry[key] for entry in entries) for key in entries[0] if key != "source_components_by_size"}
        totals[lesion]["source_components_by_size"] = {
            size: {key: sum(entry["source_components_by_size"][size][key] for entry in entries)
                   for key in ("source_components", "sampled_components", "lost_components")}
            for size in ("tiny", "small", "medium", "large")}
    report = {"validation_samples": len(records), "resized_shape": list(dataset.image_size),
              "sampling_grid_matches_dataset": True, "totals": totals, "records": records,
              "component_definition": "8-connected source annotations; size bins normalized by image area as existing diagnostics",
              "lost_definition": "no pixel of a source component selected by the actual nearest-neighbor resize grid",
              "scope": "read-only preprocessing audit; raster regions are not clinical instances; no model recall or accuracy claim"}
    atomic_json(output, report)
    print(json.dumps({key: report[key] for key in ("validation_samples", "resized_shape", "totals")}, indent=2))


if __name__ == "__main__":
    main()
