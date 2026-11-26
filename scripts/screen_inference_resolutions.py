"""Screen smaller inference sizes at the original resolution and fixed thresholds."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from bs.inference_runtime import load_model, make_dataset, sha256, write_json
from bs.inference_optimizations import batch_ssm_directions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--fold", default="f1")
    p.add_argument("--sizes", type=int, nargs="+", default=[736, 704, 672, 640])
    p.add_argument("--reference-dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location(
        "_quality", PROJECT_ROOT / "scripts/evaluate_inference_runtime.py"
    )
    quality_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quality_module)
    torch.set_num_threads(2)
    model, config = load_model(args.checkpoint)
    batch_ssm_directions(model)
    model.to("cuda", memory_format=torch.channels_last)
    dataset = make_dataset(config, args.fold)
    manifest = json.loads((Path(args.reference_dir) / "manifest.json").read_text())
    if (
        manifest["checkpoint_sha256"] != sha256(Path(args.checkpoint))
        or manifest["fold"] != args.fold
        or manifest["sample_ids"] != [s.sample_id for s in dataset.samples]
    ):
        raise ValueError("Reference checkpoint, fold or sample order do not match")
    reference = np.load(Path(args.reference_dir) / "logits.npy", mmap_mode="r")
    results = []
    for size in args.sizes:
        quality = quality_module.QualityAccumulator()
        started = time.perf_counter()
        for i in range(len(dataset)):
            sample = dataset[i]
            inputs = sample["image"][None].to("cuda", memory_format=torch.channels_last)
            inputs = F.interpolate(
                inputs, size=(size, size), mode="bilinear", align_corners=False
            )
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                logits = model(inputs)
            logits = (
                F.interpolate(
                    logits.float(),
                    size=sample["mask"].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )[0]
                .cpu()
                .numpy()
            )
            quality.update(
                sample["sample_id"],
                logits,
                sample["mask"].numpy(),
                manifest["thresholds"],
                reference[i],
            )
        metrics = quality.compute()
        baseline = manifest["quality"]
        delta = {
            k: metrics[k] - baseline[k]
            for k in [
                "macro_dice",
                "dice_1",
                "dice_2",
                "missed_images_1",
                "missed_images_2",
            ]
        }
        passed = delta["macro_dice"] >= -0.002 and all(
            delta[f"dice_{c}"] >= -0.003 and delta[f"missed_images_{c}"] <= 0
            for c in [1, 2]
        )
        result = {
            "size": size,
            "quality": metrics,
            "delta": delta,
            "accuracy_gate_passed": passed,
            "elapsed_seconds": time.perf_counter() - started,
        }
        results.append(result)
        write_json(args.output, results)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
