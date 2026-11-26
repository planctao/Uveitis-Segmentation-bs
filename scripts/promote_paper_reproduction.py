"""After verified paper screening, run matched-resolution FA comparisons.

Select two complete architectures on f1 at 256px, compare both at the existing
768px resolution, then run the best paper architecture and the nnU-Net control
on the remaining four folds. This is exploratory model selection using the FA
folds; five-fold summaries are not an untouched external-test estimate.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_paper_reproduction import PROJECT_ROOT, atomic_json, process_alive, run_plan
from scripts.verify_paper_experiment import SUMMARY_FIELDS, verify_run

METHODS = ["condseg", "pbeunet", "medcagd", "udmamba", "spectralmamba", "puunet", "ukan", "mkunet"]


def summarize_fivefold_metrics(rows: list[dict]) -> dict:
    """Report both lesions and macro under each threshold policy."""
    if len(rows) != 5 or {row.get("fold") for row in rows} != {f"f{i}" for i in range(1, 6)}:
        raise ValueError("Five-fold statistics require one verified row for each of f1-f5")
    fields = [field for field in SUMMARY_FIELDS if field != "best_val_loss"]
    return {field: {"mean": statistics.mean(float(row[field]) for row in rows),
                    "sd": statistics.stdev(float(row[field]) for row in rows)}
            for field in fields}


def audit_completed_queue(root: Path, state: dict, phase: str) -> None:
    """Promotion requires checkpoint evidence, including same-epoch scores."""
    if state.get("status") != "complete" or not state.get("jobs"):
        raise RuntimeError(f"Cannot audit an incomplete or empty {phase} queue")
    for name, job in state["jobs"].items():
        if job.get("status") != "complete":
            raise RuntimeError(f"Cannot promote incomplete {phase} job {name}")
        report = verify_run(Path(job["run"]), job["fold"], int(job["epochs"]), require_full_data=True)
        if job.get("metrics") != report["metrics"]:
            raise RuntimeError(f"{phase} queue score differs from checkpoint audit: {name}")
        atomic_json(root / "verified_experiments" / f"{name}.json", {"phase": phase, **report})


def make_jobs(method: str, folds: list[str], resolution: int) -> list[dict]:
    jobs = []
    for fold in folds:
        name = f"author_{method}_{fold}_30_{resolution}"
        override = {"train": {"image_size": [resolution, resolution], "batch_size": 1,
                              "grad_accum_steps": 8},
                    "reproduction": {"experiment_class": "matched_resolution" if fold == "f1" else "fivefold",
                                     "input_resolution": resolution,
                                     "training_protocol": "FA full-data 30 epochs at baseline resolution; paper-specific loss/initialization"}}
        job = {"id": name, "config": f"configs/paper_reproduction/{method}.yaml", "fold": fold, "overrides": override}
        if method == "condseg":
            stage1 = {"id": f"author_condseg_cr_{fold}_30_{resolution}",
                      "config": "configs/paper_reproduction/condseg_cr.yaml", "fold": fold,
                      "overrides": override}
            jobs.append(stage1)
            job["stage1_job"] = stage1["id"]
        jobs.append(job)
    return jobs


def control_jobs(folds: list[str], resolution: int) -> list[dict]:
    return [{"id": f"author_nnunet_control_{fold}_30_{resolution}",
             "config": f"configs/paper_reproduction/nnunet_control_{resolution}.yaml", "fold": fold}
            for fold in folds]


def run_matched_plan(root: Path, plan: Path, resume: bool) -> None:
    """Observe an existing control worker instead of launching duplicate runs."""
    control_plan, control_state = root / "control_precompute_plan.json", root / "control_precompute_state.json"
    if control_plan.exists() != control_state.exists():
        raise RuntimeError("Precomputed control ownership requires both its plan and state")
    run_plan(plan, root / "matched_resolution_state.json", resume,
             external_plan=control_plan if control_plan.exists() else None,
             external_state=control_state if control_plan.exists() else None)


def promote(root: Path, resume: bool) -> None:
    status_path = root / "promotion_status.json"
    atomic_json(status_path, {"status": "waiting_for_verified_screen", "supervisor_pid": os.getpid()})
    screen_path = root / "screen_state.json"
    while True:
        if not screen_path.exists():
            time.sleep(5)
            continue
        screen = json.loads(screen_path.read_text())
        if screen["status"] == "complete":
            break
        if screen["status"] == "failed":
            atomic_json(status_path, {"status": "screen_failed_requires_diagnosis"})
            raise RuntimeError("Screen queue failed; diagnose before promotion")
        if not process_alive(screen["supervisor_pid"], screen.get("supervisor_start")):
            raise RuntimeError("Screen supervisor disappeared; recover it before promotion")
        time.sleep(20)
    audit_completed_queue(root, screen, "screen")
    rank = []
    for method in METHODS:
        row = screen["jobs"][f"author_{method}_f1_30_256"]
        if row["status"] != "complete":
            raise RuntimeError(f"Incomplete method cannot be selected: {method}")
        rank.append((float(row["metrics"]["best_paper_sweep_macro_dice"]), method))
    chosen = [method for _, method in sorted(rank, reverse=True)[:2]]
    atomic_json(status_path, {"status": "matched_resolution_running", "selected_256": chosen,
                              "selection_metric": "shared threshold macro at same selected checkpoint",
                              "screen_scores": rank, "supervisor_pid": os.getpid()})
    jobs = control_jobs(["f1"], 256) + control_jobs(["f1"], 768)
    for method in chosen:
        jobs += make_jobs(method, ["f1"], 768)
    high_plan = root / "matched_resolution_plan.json"
    if not high_plan.exists():
        atomic_json(high_plan, {"jobs": jobs})
    run_matched_plan(root, high_plan, resume)
    high = json.loads((root / "matched_resolution_state.json").read_text())
    audit_completed_queue(root, high, "matched_resolution")
    winner = max(chosen, key=lambda m: float(high["jobs"][f"author_{m}_f1_30_768"]["metrics"]["best_paper_sweep_macro_dice"]))
    control = high["jobs"]["author_nnunet_control_f1_30_768"]["metrics"]
    winner_metrics = high["jobs"][f"author_{winner}_f1_30_768"]["metrics"]
    atomic_json(status_path, {"status": "remaining_folds_running", "selected_256": chosen,
                              "winner_768": winner, "winner_f1_metrics": winner_metrics,
                              "control_f1_metrics": control, "supervisor_pid": os.getpid()})
    remaining = ["f2", "f3", "f4", "f5"]
    jobs = make_jobs(winner, remaining, 768) + control_jobs(remaining, 768)
    fold_plan = root / "fivefold_plan.json"
    if not fold_plan.exists():
        atomic_json(fold_plan, {"jobs": jobs})
    run_plan(fold_plan, root / "fivefold_state.json", resume)
    last = json.loads((root / "fivefold_state.json").read_text())
    audit_completed_queue(root, last, "fivefold")
    score_rows = {"paper": [winner_metrics], "control": [control]}
    for fold in remaining:
        score_rows["paper"].append(last["jobs"][f"author_{winner}_{fold}_30_768"]["metrics"])
        score_rows["control"].append(last["jobs"][f"author_nnunet_control_{fold}_30_768"]["metrics"])
    report = {"winner": winner, "resolution": 768, "epochs": 30, "rows": score_rows,
              "limitation": "Architecture and thresholds selected using FA validation; no untouched test set"}
    for name, rows in score_rows.items():
        report[name] = summarize_fivefold_metrics(rows)
    atomic_json(root / "fivefold_comparison.json", report)
    atomic_json(status_path, {"status": "complete", "winner_768": winner, "report": str(root / "fivefold_comparison.json")})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/paper_reproduction_20261011")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    root = Path(args.root).resolve()
    with (root / "promotion.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            promote(root, args.resume)
        except Exception as error:
            status_path = root / "promotion_status.json"
            previous = json.loads(status_path.read_text()) if status_path.exists() else {}
            atomic_json(status_path, {**previous, "status": "failed", "error": str(error), "supervisor_pid": os.getpid()})
            raise


if __name__ == "__main__":
    main()
