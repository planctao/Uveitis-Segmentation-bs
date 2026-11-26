"""Calibrate CPU INT8 on clean training folds, retaining sensitive float branches."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch

from bs.inference_runtime import make_dataset, sha256, write_json
from bs.paths import project_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--scope", choices=["all", "backbone", "decoder"], default="all")
    p.add_argument("--calibration-size", type=int, default=128)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--protect-first-stage", action="store_true")
    args = p.parse_args()
    if args.calibration_size < 1 or args.threads < 1:
        p.error("calibration-size and threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    # NNCF constructs its own OpenVINO engine. Limit visible cores so its
    # calibration thread pool respects the requested CPU budget as well.
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, available[: args.threads])
    import openvino as ov
    import nncf
    from nncf.quantization.advanced_parameters import (
        AdvancedQuantizationParameters,
        OverflowFix,
    )

    source = project_path(args.model)
    metadata = json.loads(Path(str(source) + ".json").read_text())
    dataset = make_dataset(metadata["config"], metadata["fold"], training=True)
    indices = (
        np.random.default_rng(args.seed)
        .permutation(len(dataset))[: args.calibration_size]
        .tolist()
    )
    ids = [dataset.samples[i].sample_id for i in indices]
    metadata.update(
        {
            "quantization": "INT8",
            "quantization_scope": args.scope,
            "calibration_size": len(indices),
            "calibration_sample_ids": ids,
            "calibration_folds": sorted({dataset.samples[i].fold for i in indices}),
            "calibration_validation_excluded": True,
            "calibration_seed": args.seed,
        }
    )
    if metadata["fold"] in metadata["calibration_folds"]:
        raise RuntimeError("Validation fold leaked into calibration")
    write_json(str(args.output) + ".pending.json", metadata)
    patterns = [
        r".*decode_head\.attention\.global_branch.*",
        r".*decode_head\.fuse\.3/.*",
    ]
    if args.scope == "backbone":
        patterns = [r".*decode_head.*"]
    if args.scope == "decoder":
        patterns.append(r".*backbone.*")
    if args.protect_first_stage:
        patterns.extend(
            [r".*backbone\.stages\.0\..*", r".*backbone\.downsample_layers\.0\..*"]
        )
    metadata["ignored_patterns"] = patterns
    calibration = nncf.Dataset(
        indices, lambda i: dataset[i]["image"].unsqueeze(0).numpy()
    )
    model = ov.Core().read_model(str(source))
    started = time.perf_counter()
    quantized = nncf.quantize(
        model,
        calibration,
        preset=nncf.QuantizationPreset.MIXED,
        target_device=nncf.TargetDevice.CPU,
        subset_size=len(indices),
        ignored_scope=nncf.IgnoredScope(patterns=patterns),
        advanced_parameters=AdvancedQuantizationParameters(
            overflow_fix=OverflowFix.DISABLE
        ),
    )
    ov.save_model(quantized, str(project_path(args.output)), compress_to_fp16=False)
    metadata.update(
        {
            "quantization_seconds": time.perf_counter() - started,
            "artifact_sha256": sha256(project_path(args.output)),
            "weights_sha256": sha256(project_path(args.output).with_suffix(".bin")),
            "nncf_version": nncf.__version__,
            "openvino_version": ov.__version__,
        }
    )
    write_json(str(args.output) + ".json", metadata)
    print(
        json.dumps(
            {
                k: v
                for k, v in metadata.items()
                if k not in ["config", "calibration_sample_ids"]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
