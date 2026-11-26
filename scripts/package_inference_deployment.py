"""Package a CPU artifact only after a full-fold accuracy gate has passed."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bs.inference_runtime import sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    evaluation = json.loads(Path(args.evaluation).read_text())
    timing = json.loads(Path(args.benchmark).read_text())
    if not evaluation.get("full_fold") or not evaluation.get("accuracy_gate", {}).get(
        "passed"
    ):
        p.error("Packaging requires a successful full-fold accuracy comparison")
    if evaluation["backend"] != "openvino":
        p.error("CPU deployment requires an OpenVINO artifact")
    if any(
        evaluation["quality"].get(f"new_missed_images_{c}", -1) != 0 for c in [1, 2]
    ):
        p.error("Packaging requires a paired audit with no newly missed images")
    source = Path(evaluation["artifact"])
    artifact_metadata = json.loads(Path(str(source) + ".json").read_text())
    if (
        timing.get("backend") != "openvino"
        or timing.get("device") != "cpu"
        or timing.get("threads") != evaluation["threads"]
        or timing.get("checkpoint_sha256") != evaluation["checkpoint_sha256"]
        or timing.get("artifact_sha256") != evaluation["artifact_sha256"]
        or timing.get("weights_sha256") != evaluation["weights_sha256"]
    ):
        p.error(
            "Benchmark and accuracy evaluation must use the same artifact and threads"
        )
    if evaluation["artifact_sha256"] != sha256(source) or evaluation[
        "weights_sha256"
    ] != sha256(source.with_suffix(".bin")):
        p.error("Artifact changed after validation")
    config = evaluation["config"]
    if config.get("preprocess", {}).get("enabled", False):
        p.error("This NumPy deployment does not implement custom FA preprocessing")
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for suffix in [".xml", ".bin"]:
        shutil.copy2(source.with_suffix(suffix), destination / ("model" + suffix))
    metadata = {
        "format_version": 1,
        "model": "model.xml",
        "input_size": config["train"]["image_size"],
        "thresholds": evaluation["thresholds"],
        "threads": timing["threads"],
        "quantization": evaluation.get("quantization"),
        "quantization_scope": artifact_metadata.get("quantization_scope"),
        "batched_ssm_directions": artifact_metadata.get(
            "batched_ssm_directions", False
        ),
        "calibration_folds": artifact_metadata.get("calibration_folds"),
        "calibration_validation_excluded": artifact_metadata.get(
            "calibration_validation_excluded"
        ),
        "checkpoint_sha256": evaluation["checkpoint_sha256"],
        "artifact_sha256": evaluation["artifact_sha256"],
        "weights_sha256": evaluation["weights_sha256"],
        "validation_fold": evaluation["fold"],
        "validation_samples": evaluation["quality"]["samples"],
        "quality": evaluation["quality"],
        "reference_quality": evaluation["reference_quality"],
        "delta": evaluation["delta"],
        "accuracy_gate": evaluation["accuracy_gate"],
        "benchmark": timing,
        "mask_labels": {
            "0": "background",
            "1": "lesion_1",
            "2": "lesion_2",
            "3": "overlap",
        },
    }
    write_json(destination / "deployment.json", metadata)
    print(
        json.dumps(
            {
                "deployment": str(destination / "deployment.json"),
                "quality": evaluation["quality"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
