"""Run the validated CPU deployment on one image or a directory of images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
from PIL import Image

from bs.cpu_predictor import CPUSegmenter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deployment", required=True, help="Validated deployment.json")
    p.add_argument("--input", required=True, help="Image file or directory")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--threads", type=int)
    args = p.parse_args()
    source, destination = Path(args.input), Path(args.output_dir)
    if source.is_dir():
        files = sorted(
            f
            for f in source.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".bmp"}
        )
    else:
        files = [source]
    if not files:
        p.error("No input images")
    destination.mkdir(parents=True, exist_ok=True)
    segmenter = CPUSegmenter(args.deployment, threads=args.threads)
    palette = np.array(
        [[0, 0, 0], [255, 64, 64], [64, 210, 110], [80, 150, 255]], dtype=np.uint8
    )
    rows = []
    for source_file in files:
        with Image.open(source_file) as image:
            result = segmenter.predict(image)
        Image.fromarray(result["labels"]).save(
            destination / f"{source_file.stem}_labels.png"
        )
        Image.fromarray(palette[result["labels"]]).save(
            destination / f"{source_file.stem}_mask_rgb.png"
        )
        row = {
            "image": str(source_file),
            "timing_ms": result["timing_ms"],
            "lesion_1_pixels": int(result["lesion_1"].sum()),
            "lesion_2_pixels": int(result["lesion_2"].sum()),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    (destination / "predictions.json").write_text(
        json.dumps({"model_load_ms": segmenter.load_ms, "predictions": rows}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
