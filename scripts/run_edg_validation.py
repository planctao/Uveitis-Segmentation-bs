"""Run the immutable three-seed EDG fusion screen and verify actual checkpoints."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.run_paper_reproduction import atomic_json, process_start, run_plan
from scripts.verify_paper_experiment import verify_run

SEEDS = (42, 2026, 3407)
METHODS = {"concat": "nnunet_2d_multilabel", "equal": "difference_equal",
           "fdg": "difference_fdg", "edg": "difference_edg"}


def prepare_plan(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    configs = root / "input_configs"
    configs.mkdir(exist_ok=True)
    base_path = PROJECT_ROOT / "outputs/paper_reproduction_20261011/queue_configs/author_edg_f1_30_256.yaml"
    base = yaml.safe_load(base_path.read_text())
    jobs = []
    for seed in SEEDS:
        for method, model_name in METHODS.items():
            name = f"paired_{method}_s{seed}_f1_30_256"
            config = json.loads(json.dumps(base))
            config["project"].update(name=name, seed=seed)
            config["runtime"].update(num_workers=2, paired_model_seed=seed, paired_data_seed=seed)
            config["model"]["name"] = model_name
            config["outputs"]["root"] = str(root.relative_to(PROJECT_ROOT) / "runs")
            config["reproduction"] = {
                "experiment_class": "paired_fusion_ablation", "paper_method": method,
                "implementation": "controlled_nnunet_fusion", "input_resolution": 256,
                "training_protocol": "f1 clean full validation; 30 epochs; batch8; paired seeds and epoch data streams",
                "deviation": "same nnU-Net backbone; equal control retains all FE parameters but statistic branches do not affect output",
            }
            path = configs / f"{name}.yaml"
            content = yaml.safe_dump(config, sort_keys=False)
            if path.exists() and path.read_text() != content:
                raise RuntimeError(f"Immutable experiment config changed: {path}; use another experiment root")
            if not path.exists():
                path.write_text(content)
            jobs.append({"id": name, "config": str(path.relative_to(PROJECT_ROOT)), "fold": "f1"})
    plan_path = root / "paired_plan.json"
    plan = {"seeds": list(SEEDS), "methods": list(METHODS), "jobs": jobs}
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError("Plan differs from existing immutable experiment")
    if not plan_path.exists():
        atomic_json(plan_path, plan)
    return plan_path


def compare_initial_weights(runs: dict[str, Path], seed: int) -> dict:
    fingerprints = {method: json.loads((run / "f1/initial_weight_fingerprints.json").read_text())
                    for method, run in runs.items()}
    if not (fingerprints["equal"] == fingerprints["fdg"] == fingerprints["edg"]):
        raise RuntimeError(f"Fusion initial weights were not paired for seed {seed}")
    common = [name for name, item in fingerprints["concat"].items()
              if name in fingerprints["edg"] and item["shape"] == fingerprints["edg"][name]["shape"]]
    if not common or any(fingerprints["concat"][name] != fingerprints["edg"][name] for name in common):
        raise RuntimeError(f"Shared concat and fusion initial weights differ for seed {seed}")
    return {"seed": seed, "fusion_initial_states_identical": True, "matched_concat_state_tensors": len(common),
            "unmatched_concat_tensors": [name for name in fingerprints["concat"] if name not in common]}


def aggregate(root: Path, state: dict) -> dict:
    rows, checks = [], []
    for seed in SEEDS:
        runs = {method: Path(state["jobs"][f"paired_{method}_s{seed}_f1_30_256"]["run"]) for method in METHODS}
        checks.append(compare_initial_weights(runs, seed))
        for method, run in runs.items():
            summary = json.loads((run / "summary.json").read_text())["folds"][0]
            rows.append({"seed": seed, "method": method, **summary})
    comparisons = {}
    for control in ("concat", "equal", "fdg"):
        comparisons[control] = {}
        for label, key in (("shared", "best_paper_sweep_macro_dice"), ("fixed_0.5", "best_paper_macro_dice")):
            deltas = []
            for seed in SEEDS:
                selected = {row["method"]: row for row in rows if row["seed"] == seed}
                deltas.append(selected["edg"][key] - selected[control][key])
            comparisons[control][label] = {
                "paired_deltas": deltas, "mean_delta": float(np.mean(deltas)),
                "sample_std": float(np.std(deltas, ddof=1)), "all_positive": all(delta > 0 for delta in deltas),
            }
    # This decision schedules further validation; it is not a significance
    # claim. Three seeds on one validation fold remain exploratory.
    advance = all(comparisons[control]["shared"]["all_positive"] for control in ("concat", "equal"))
    report = {"rows": rows, "initialization_checks": checks, "edg_minus_control": comparisons,
              "advance_to_placement_and_matched_768": advance,
              "decision_rule": "EDG shared-threshold macro must exceed both concat and equal in every paired seed",
              "scope": "f1/256 exploration; selected checkpoints and thresholds use the same validation fold; no held-out gain claimed"}
    atomic_json(root / "paired_results.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/edg_validation_20261011")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = (PROJECT_ROOT / args.root).resolve()
    root.relative_to(PROJECT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = prepare_plan(root)
        if args.prepare_only:
            print(plan_path)
            return
        status_path, queue_state = root / "validation_status.json", root / "paired_queue_state.json"
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        if previous.get("status") == "screen_complete":
            print("Paired screen and audits already complete")
            return
        source_paths = ["src/bs/recent_unet_methods.py", "src/bs/paired_experiments.py", "scripts/train_unet_multilabel.py"]
        hashes = {path: hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest() for path in source_paths}
        if previous.get("source_sha256", hashes) != hashes:
            raise RuntimeError("Training implementation changed since experiment start; inspect before recovery")
        status = {"status": "paired_screen_running", "supervisor_pid": os.getpid(),
                  "supervisor_start": process_start(os.getpid()), "source_sha256": hashes,
                  "started_at": previous.get("started_at", time.time()), "plan": str(plan_path)}
        atomic_json(status_path, status)
        try:
            run_plan(plan_path, queue_state, resume=args.resume)
            state = json.loads(queue_state.read_text())
            atomic_json(status_path, {**status, "status": "auditing_checkpoints"})
            audits = root / "verified_experiments"
            audits.mkdir(exist_ok=True)
            for job in json.loads(plan_path.read_text())["jobs"]:
                entry = state["jobs"][job["id"]]
                report = verify_run(Path(entry["run"]), "f1", 30, require_full_data=True)
                atomic_json(audits / f"{job['id']}.json", report)
                replay_path = audits / f"{job['id']}_replay.json"
                if not replay_path.exists():
                    subprocess.run([sys.executable, "scripts/audit_paper_checkpoint_evaluation.py", "--run", entry["run"],
                                    "--output", str(replay_path)], cwd=PROJECT_ROOT, check=True)
                replay = json.loads(replay_path.read_text())
                if not replay.get("all_required_metrics_match") or replay.get("validation_samples") != 444:
                    raise RuntimeError(f"Checkpoint replay failed: {replay_path}")
            results = aggregate(root, state)
            atomic_json(status_path, {**status, "status": "screen_complete", "finished_at": time.time(),
                                     "results": str(root / "paired_results.json"),
                                     "advance_to_placement_and_matched_768": results["advance_to_placement_and_matched_768"]})
        except Exception as error:
            atomic_json(status_path, {**status, "status": "failed", "error": str(error)})
            raise


if __name__ == "__main__":
    main()
