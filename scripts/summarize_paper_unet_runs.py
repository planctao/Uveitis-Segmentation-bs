"""Report paper methods without ranking across resolutions or training protocols.

Dice columns refer to the SAME selected checkpoint in a fold. Validation
threshold sweeps are exploratory. Smoke runs are excluded by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_paper_reproduction import completed_metrics


def load_run(path: Path) -> dict[str, object] | None:
    config_path = path / "resolved_config.json"
    if not config_path.exists():
        return None
    config = json.loads(config_path.read_text())
    metadata = config.get("reproduction", {})
    epochs = int(config["train"]["epochs"])
    limited = bool(config["train"].get("max_train_samples") or config["train"].get("max_val_samples"))
    category = "smoke" if limited else metadata.get("experiment_class", "exploratory_ablation")
    stage1 = config["model"].get("name") == "condseg_author_stage1"
    if stage1 and category != "smoke":
        category = "auxiliary_stage1"
    folds = config["train"]["folds_to_run"]
    valid = [r for fold in folds if (r := completed_metrics(path, fold, epochs)) is not None]
    result: dict[str, object] = {
        "run": path.name, "model": config["model"].get("name", ""), "category": category,
        "implementation": metadata.get("implementation", "simplified_module_adaptation"),
        "resolution": "x".join(str(v) for v in config["train"]["image_size"]),
        "epochs": epochs, "folds": "/".join(folds), "completed_folds": len(valid),
        "stage": "stage1" if stage1 else "single_stage_or_stage2",
        "candidate_result": category == "screen" and not stage1 and len(valid) == len(folds),
        "pretrained": config["model"].get("pretrained", False), "loss": config["loss"]["name"],
        "status": "complete" if len(valid) == len(folds) else "incomplete_or_invalid",
    }
    for label, field in [("fixed_dice1", "best_paper_dice_1"), ("fixed_dice2", "best_paper_dice_2"),
                         ("fixed_macro", "best_paper_macro_dice"), ("shared_dice1", "best_paper_sweep_dice_1"),
                         ("shared_dice2", "best_paper_sweep_dice_2"), ("shared_macro", "best_paper_sweep_macro_dice"),
                         ("ind_dice1", "best_paper_sweep_ind_dice_1"), ("ind_dice2", "best_paper_sweep_ind_dice_2"),
                         ("ind_macro", "best_paper_sweep_ind_macro_dice")]:
        values = [float(r[field]) for r in valid if field in r]
        result[label] = sum(values) / len(values) if values else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", default=None)
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows, seen = [], set()
    for root in args.root or ["outputs/paper_reproduction_20261011/runs", "runs"]:
        if not Path(root).exists():
            continue
        for path in sorted(Path(root).iterdir()):
            if not path.is_dir() or path.resolve() in seen or not path.name.startswith(("paper", "author", "nnunet")):
                continue
            seen.add(path.resolve())
            row = load_run(path)
            if row and (args.include_smoke or row["category"] != "smoke"):
                rows.append(row)
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False))
    elif rows:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
