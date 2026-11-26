"""Measure the actual CPU-only prediction path, including preprocessing and masks."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
from PIL import Image

from bs.cpu_predictor import CPUSegmenter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deployment", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--threads", type=int)
    p.add_argument("--iterations", type=int, default=50)
    args = p.parse_args()
    if args.iterations < 1:
        p.error("iterations must be positive")
    started = time.perf_counter()
    engine = CPUSegmenter(args.deployment, threads=args.threads)
    load_ms = (time.perf_counter() - started) * 1000
    with Image.open(args.image) as file:
        image = file.convert("RGB")
    first = engine.predict(image)["timing_ms"]
    for _ in range(3):
        engine.predict(image)
    timings = [engine.predict(image)["timing_ms"] for _ in range(args.iterations)]
    result = {
        "deployment": str(Path(args.deployment).resolve()),
        "platform": platform.platform(),
        "threads": args.threads or engine.metadata["threads"],
        "load_ms": load_ms,
        "first_prediction_ms": first,
        "iterations": args.iterations,
        "timing_ms": {
            key: {
                "mean": float(np.mean([t[key] for t in timings])),
                "p50": float(np.percentile([t[key] for t in timings], 50)),
                "p95": float(np.percentile([t[key] for t in timings], 95)),
            }
            for key in timings[0]
        },
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
        "torch_imported": "torch" in sys.modules,
        "execution_devices": engine.compiled.get_property("EXECUTION_DEVICES"),
        "scope": "decoded PIL image to multilabel masks; file decoding/writing and model load excluded from warm timings",
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
