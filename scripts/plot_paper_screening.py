"""Plot actual f1/256 screening trajectories, keeping unfinished runs explicit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_paper_reproduction import atomic_json, completed_metrics


METHODS = [
    ("condseg", "ConDSeg stage 2"), ("pbeunet", "PBE-UNet"),
    ("medcagd", "MedCAGD"), ("udmamba", "UD-Mamba"),
    ("spectralmamba", "SpectralMamba"), ("puunet", "PU-UNet"),
    ("ukan", "U-KAN"), ("mkunet", "MK-UNet"),
]
FIELDS = [("val_paper_dice_1", "Lesion 1"), ("val_paper_dice_2", "Lesion 2"),
          ("val_paper_macro_dice", "Macro Dice")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/paper_reproduction_20261011")
    args = parser.parse_args()
    root = Path(args.root)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharex=True, sharey=True)
    included = []
    for index, (method, label) in enumerate(METHODS):
        run = root / "runs" / f"author_{method}_f1_30_256"
        config_path, metric_path = run / "resolved_config.json", run / "f1/metrics.csv"
        if not config_path.exists() or not metric_path.exists():
            continue
        config_bytes, metric_bytes = config_path.read_bytes(), metric_path.read_bytes()
        config = json.loads(config_bytes)
        if (config["train"]["image_size"] != [256, 256]
                or config["train"]["folds_to_run"] != ["f1"]
                or config["train"].get("max_train_samples") or config["train"].get("max_val_samples")
                or config["metric"]["threshold"] != .5 or not config["data"]["exclude_val_augmented"]):
            raise ValueError(f"Unexpected plot protocol: {run}")
        rows = list(csv.DictReader(io.StringIO(metric_bytes.decode())))
        if not rows:
            continue
        epochs = [int(row["epoch"]) for row in rows]
        if epochs != list(range(1, len(rows) + 1)) or len(rows) > 30:
            raise ValueError(f"Incomplete or inconsistent epoch history: {run}")
        complete = len(rows) == 30 and completed_metrics(run, "f1", 30) is not None
        for ax, (field, title) in zip(axes, FIELDS):
            values = [float(row[field]) for row in rows]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Nonfinite curve: {run} {field}")
            ax.plot(epochs, values, label=label + (" (in progress)" if not complete else ""),
                    color=plt.get_cmap("tab10")(index), linewidth=1.7,
                    linestyle="-" if complete else "--")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_xlim(1, 30)
            ax.set_ylim(0, .9)
            ax.grid(alpha=.2)
        included.append({"method": label, "run": str(run), "epochs_observed": len(rows),
                         "complete_history_and_summary": complete,
                         "resolved_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                         "metric_history_sha256": hashlib.sha256(metric_bytes).hexdigest()})
    if not included:
        raise ValueError("No formal f1/256 screening history to plot")
    axes[0].set_ylabel("Validation Dice, fixed threshold 0.5")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("FA screening on f1 at 256 px: training trajectories", fontsize=12)
    fig.text(.5, .015,
             "1965 training / 444 clean validation images; paper-specific losses and initialization. "
             "In-progress curves are not final results.", ha="center", fontsize=8)
    fig.tight_layout(rect=[0, .06, 1, .95])
    output = root / "figures/screen_fixed_threshold_learning_curves"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    report = {"generated_at": time.time(), "included": included, "CR_excluded_from_candidate_plot": True,
              "scope": "Fixed0.5 validation trajectories at f1/256; differing paper losses/initializations. "
                       "Curve completion is not a substitute for actual checkpoint/replay verification."}
    atomic_json(output.with_suffix(".json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
