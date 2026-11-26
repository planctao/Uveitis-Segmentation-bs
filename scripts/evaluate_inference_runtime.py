"""Compare deployment engines on identical clean validation images and thresholds."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch

from bs.inference_runtime import (
    ONNXEngine,
    OpenVINOEngine,
    TorchEngine,
    load_model,
    make_dataset,
    sha256,
    write_json,
)
from bs.paths import project_path


class QualityAccumulator:
    def __init__(self):
        self.intersection = np.zeros(2, dtype=np.int64)
        self.predicted = np.zeros(2, dtype=np.int64)
        self.target = np.zeros(2, dtype=np.int64)
        self.rows = []

    def update(self, sample_id, logits, mask, thresholds, reference=None):
        if not np.isfinite(logits).all():
            raise RuntimeError(f"Non-finite predictions for {sample_id}")
        valid = mask != 255
        targets = (
            np.stack([(mask == 1) | (mask == 3), (mask == 2) | (mask == 3)]) & valid
        )
        threshold_logits = np.log(
            np.asarray(thresholds) / (1.0 - np.asarray(thresholds))
        )[:, None, None]
        pred = (logits >= threshold_logits) & valid
        intersection = (pred & targets).sum(axis=(1, 2))
        predicted = pred.sum(axis=(1, 2))
        target = targets.sum(axis=(1, 2))
        self.intersection += intersection
        self.predicted += predicted
        self.target += target
        row = {"sample_id": sample_id}
        if reference is not None:
            reference_pred = (reference >= threshold_logits) & valid
            row["mask_disagreement_pixels"] = int((pred != reference_pred).sum())
            row["max_logit_difference"] = float(np.abs(logits - reference).max())
        for c in range(2):
            denom = predicted[c] + target[c]
            row[f"dice_{c + 1}"] = float(2 * intersection[c] / denom) if denom else 1.0
            row[f"target_pixels_{c + 1}"] = int(target[c])
            row[f"predicted_pixels_{c + 1}"] = int(predicted[c])
            row[f"area_error_{c + 1}"] = float(
                (int(predicted[c]) - int(target[c])) / max(1, valid.sum())
            )
            row[f"missed_image_{c + 1}"] = int(target[c] > 0 and predicted[c] == 0)
            if reference is not None:
                row[f"new_missed_image_{c + 1}"] = int(
                    target[c] > 0 and predicted[c] == 0 and reference_pred[c].any()
                )
            row[f"false_positive_image_{c + 1}"] = int(
                target[c] == 0 and predicted[c] > 0
            )
        self.rows.append(row)

    def compute(self):
        dice = 2 * self.intersection / np.maximum(1, self.predicted + self.target)
        out = {
            "dice_1": float(dice[0]),
            "dice_2": float(dice[1]),
            "macro_dice": float(dice.mean()),
            "samples": len(self.rows),
        }
        for c in range(2):
            positive = [r for r in self.rows if r[f"target_pixels_{c + 1}"] > 0]
            out[f"positive_images_{c + 1}"] = len(positive)
            out[f"positive_image_mean_dice_{c + 1}"] = (
                float(np.mean([r[f"dice_{c + 1}"] for r in positive]))
                if positive
                else None
            )
            out[f"missed_images_{c + 1}"] = sum(
                r[f"missed_image_{c + 1}"] for r in self.rows
            )
            out[f"false_positive_images_{c + 1}"] = sum(
                r[f"false_positive_image_{c + 1}"] for r in self.rows
            )
            out[f"area_mae_{c + 1}"] = float(
                np.mean([abs(r[f"area_error_{c + 1}"]) for r in self.rows])
            )
        if self.rows and "mask_disagreement_pixels" in self.rows[0]:
            out["mask_disagreement_pixels"] = sum(
                r["mask_disagreement_pixels"] for r in self.rows
            )
            out["max_logit_difference"] = max(
                r["max_logit_difference"] for r in self.rows
            )
            for c in [1, 2]:
                newly_missed = [
                    r["sample_id"] for r in self.rows if r[f"new_missed_image_{c}"]
                ]
                out[f"new_missed_images_{c}"] = len(newly_missed)
                out[f"new_missed_sample_ids_{c}"] = newly_missed
        return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["reference", "compare"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--fold", default="f1")
    p.add_argument("--backend", choices=["torch", "openvino", "onnx"], default="torch")
    p.add_argument("--artifact")
    p.add_argument("--device", default="cpu")
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    p.add_argument("--layout", choices=["nchw", "channels_last"], default="nchw")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--batch-ssm", action="store_true")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--thresholds", type=float, nargs=2, default=[0.9, 0.9])
    p.add_argument("--reference-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    if not all(0 < t < 1 for t in args.thresholds):
        p.error("thresholds must lie strictly between zero and one")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    model, config = load_model(args.checkpoint)
    if args.batch_ssm:
        from bs.inference_optimizations import batch_ssm_directions

        batch_ssm_directions(model)
    dataset = make_dataset(config, args.fold)
    size = min(len(dataset), args.limit) if args.limit else len(dataset)
    reference_dir = project_path(args.reference_dir)
    reference_dir.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(project_path(args.checkpoint))
    ids = [s.sample_id for s in dataset.samples[:size]]
    metadata = {
        "checkpoint": str(project_path(args.checkpoint)),
        "checkpoint_sha256": source_hash,
        "fold": args.fold,
        "sample_ids": ids,
        "thresholds": args.thresholds,
        "backend": args.backend,
        "device": args.device,
        "precision": args.precision,
        "layout": args.layout,
        "threads": args.threads,
        "full_fold": size == len(dataset),
        "compiled": args.compile,
        "batched_ssm_directions": args.batch_ssm,
        "validation_augmented_excluded": True,
        "tta": False,
        "config": config,
    }
    if args.action == "reference":
        cache = np.lib.format.open_memmap(
            reference_dir / "logits.npy",
            mode="w+",
            dtype=np.float32,
            shape=(size, 2, *config["train"]["image_size"]),
        )
    else:
        manifest = json.loads((reference_dir / "manifest.json").read_text())
        if (
            manifest["checkpoint_sha256"] != source_hash
            or manifest["sample_ids"][:size] != ids
            or manifest["thresholds"] != args.thresholds
        ):
            raise ValueError(
                "Reference checkpoint, sample order or thresholds do not match"
            )
        cache = np.load(reference_dir / "logits.npy", mmap_mode="r")
    if args.artifact:
        artifact = project_path(args.artifact)
        artifact_metadata = json.loads(Path(str(artifact) + ".json").read_text())
        if artifact_metadata["checkpoint_sha256"] != source_hash:
            raise ValueError("Exported artifact was built from a different checkpoint")
        metadata["artifact"] = str(artifact)
        metadata["artifact_sha256"] = sha256(artifact)
        metadata["weights_sha256"] = (
            sha256(artifact.with_suffix(".bin")) if args.backend == "openvino" else None
        )
        metadata["quantization"] = artifact_metadata.get("quantization")
        metadata["quantization_scope"] = artifact_metadata.get("quantization_scope")
        metadata["batched_ssm_directions"] = artifact_metadata.get(
            "batched_ssm_directions", False
        )
        metadata["precision"] = (
            "mixed_int8_fp32" if metadata["quantization"] == "INT8" else "fp32"
        )
    if args.backend == "torch":
        engine = TorchEngine(
            model,
            device=args.device,
            precision=args.precision,
            layout=args.layout,
            compiled=args.compile,
        )
    elif args.backend == "openvino":
        engine = OpenVINOEngine(args.artifact, threads=args.threads)
    else:
        engine = ONNXEngine(args.artifact, threads=args.threads)
    quality = QualityAccumulator()
    reference_quality = QualityAccumulator()
    timings = []
    total_started = time.perf_counter()
    for i in range(size):
        sample = dataset[i]
        inputs = sample["image"].unsqueeze(0).numpy()
        started = time.perf_counter()
        logits = engine(inputs)[0]
        timings.append((time.perf_counter() - started) * 1000)
        mask = sample["mask"].numpy()
        if args.action == "reference":
            cache[i] = logits
        ref = np.asarray(cache[i]) if args.action == "compare" else None
        quality.update(sample["sample_id"], logits, mask, args.thresholds, ref)
        if ref is not None:
            reference_quality.update(sample["sample_id"], ref, mask, args.thresholds)
        if (i + 1) % 20 == 0 or i + 1 == size:
            print(
                json.dumps(
                    {
                        "completed": i + 1,
                        "total": size,
                        "elapsed_seconds": round(
                            time.perf_counter() - total_started, 1
                        ),
                        **quality.compute(),
                    }
                ),
                flush=True,
            )
    result = {
        **metadata,
        "quality": quality.compute(),
        "inference_mean_ms": float(np.mean(timings)),
        "inference_p95_ms": float(np.percentile(timings, 95)),
        "total_evaluation_seconds": time.perf_counter() - total_started,
    }
    if args.action == "reference":
        cache.flush()
        write_json(
            reference_dir / "manifest.json", {**metadata, "quality": quality.compute()}
        )
    else:
        original = reference_quality.compute()
        result["reference_quality"] = original
        result["delta"] = {
            k: result["quality"][k] - original[k]
            for k in [
                "dice_1",
                "dice_2",
                "macro_dice",
                "missed_images_1",
                "missed_images_2",
                "area_mae_1",
                "area_mae_2",
            ]
        }
        result["accuracy_gate"] = {
            "max_macro_drop": 0.002,
            "max_per_lesion_drop": 0.003,
            "passed": result["delta"]["macro_dice"] >= -0.002
            and all(
                result["delta"][f"dice_{c}"] >= -0.003
                and result["quality"][f"new_missed_images_{c}"] == 0
                for c in [1, 2]
            ),
        }
    write_json(args.output, result)
    csv_path = project_path(args.output).with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(quality.rows[0]))
        writer.writeheader()
        writer.writerows(quality.rows)


if __name__ == "__main__":
    main()
