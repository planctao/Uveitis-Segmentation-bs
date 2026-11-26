"""Audit completed FA experiments against checkpoints and selected-epoch metrics.

This verifies completed epochs, finite weights, configuration consistency and
all reported Dice columns belonging to the same selected checkpoint. It does
not establish author-architecture parity or out-of-sample generalization.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_paper_reproduction import atomic_json, completed_metrics


SUMMARY_FIELDS = {
    "best_paper_macro_dice": "paper_macro_dice",
    "best_paper_dice_1": "paper_dice_1", "best_paper_dice_2": "paper_dice_2",
    "best_paper_sweep_macro_dice": "paper_sweep_best_macro_dice",
    "best_paper_sweep_dice_1": "paper_sweep_best_dice_1",
    "best_paper_sweep_dice_2": "paper_sweep_best_dice_2",
    "best_paper_sweep_ind_macro_dice": "paper_sweep_ind_macro_dice",
    "best_paper_sweep_ind_dice_1": "paper_sweep_ind_dice_1",
    "best_paper_sweep_ind_dice_2": "paper_sweep_ind_dice_2",
    "best_val_loss": "loss",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, description):
    require(math.isfinite(float(actual)) and math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-12),
            f"Inconsistent {description}: {actual} != {expected}")


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_run(run: Path, fold: str, epochs: int | None = None, require_full_data: bool = False):
    config = json.loads((run / "resolved_config.json").read_text())
    epochs = epochs if epochs is not None else int(config["train"]["epochs"])
    require(epochs == int(config["train"]["epochs"]), "Requested epochs differ from resolved training config")
    require(fold in config["train"]["folds_to_run"], "Requested fold differs from resolved training config")
    if require_full_data:
        require(not config["train"].get("max_train_samples") and not config["train"].get("max_val_samples"),
                "Limited-data experiment cannot satisfy full-data verification")
        require(config["data"].get("exclude_val_augmented") is True, "Validation includes augmented duplicates")
    summary = completed_metrics(run, fold, epochs)
    require(summary is not None, "Missing complete finite metric history, summary, or latest checkpoint")
    require(summary.get("completed_epochs") == epochs, "Summary completed epoch count is inconsistent")
    with (run / fold / "metrics.csv").open(newline="") as stream:
        history = list(csv.DictReader(stream))
    metric = config["metric"]["selection_metric"]
    require(summary.get("selection_metric") == metric, "Summary selection metric differs from config")
    best = max(history, key=lambda row: float(row[f"val_{metric}"]))
    best_epoch, best_score = int(best["epoch"]), float(best[f"val_{metric}"])
    require(summary.get("best_epoch") == best_epoch, "Summary best epoch differs from earliest selected-metric maximum")
    for key, field in SUMMARY_FIELDS.items():
        require(key in summary and f"val_{field}" in best, f"Missing required metric {key}")
        close(summary[key], best[f"val_{field}"], f"summary {key} at selected epoch")
    for stem in ["paper", "paper_sweep", "paper_sweep_ind"]:
        close(summary[f"best_{stem}_macro_dice"],
              (float(summary[f"best_{stem}_dice_1"]) + float(summary[f"best_{stem}_dice_2"])) / 2,
              f"{stem} macro versus per-lesion Dice")
    checkpoints = {}
    for label, expected_epoch in [("best", best_epoch), ("latest", epochs)]:
        path = run / fold / "checkpoints" / f"{label}.pt"
        require(path.is_file(), f"Missing {label} checkpoint")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        require(payload.get("epoch") == expected_epoch, f"{label} checkpoint epoch is inconsistent")
        require(payload.get("best_epoch") == best_epoch, f"{label} checkpoint selected epoch is inconsistent")
        close(payload["best_score"], best_score, f"{label} checkpoint best score")
        for section in ["model", "data", "loss", "train", "metric", "reproduction"]:
            require(payload["config"].get(section) == config.get(section), f"{label} checkpoint {section} config differs")
        for field in SUMMARY_FIELDS.values():
            close(payload["best_metrics"][field], best[f"val_{field}"], f"{label} checkpoint selected {field}")
        state = payload["model"]
        require(bool(state), f"Empty {label} model state")
        require(all(isinstance(tensor, torch.Tensor) and torch.isfinite(tensor).all() for tensor in state.values()),
                f"Non-finite or invalid {label} model tensors")
        checkpoints[label] = {"epoch": expected_epoch, "model_tensors": len(state), "bytes": path.stat().st_size,
                              "sha256": file_sha256(path), "finite_model_state": True}
        del payload, state
        gc.collect()
    return {"run": str(run.resolve()), "fold": fold, "epochs": epochs, "best_epoch": best_epoch,
            "selection_metric": metric, "best_score": best_score, "metrics": summary,
            "metric_history_sha256": file_sha256(run / fold / "metrics.csv"),
            "resolved_config_sha256": file_sha256(run / "resolved_config.json"),
            "full_data_config_verified": require_full_data, "checkpoints": checkpoints,
            "scope": "checkpoint numerics, epoch coverage and selected-metric consistency; architecture parity is a separate audit"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--fold", default="f1")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--require-full-data", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = verify_run(Path(args.run), args.fold, args.epochs, args.require_full_data)
    if args.output:
        atomic_json(Path(args.output), report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
