"""Collect extra evidence only after an overnight job's full replay finishes.

This observer never selects a checkpoint/threshold, starts training, or edits a
training recipe. Existing manual reports may be registered in its state file.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_instance_detection import audit as instance_audit
from scripts.audit_mechanism_image_influence import compare as influence_compare
from scripts.audit_optimizer_update_counts import audit as optimizer_audit
from scripts.run_paper_reproduction import atomic_json, process_start


def read_json(path: Path):
    return json.loads(path.read_text())


def metric_rows(run: Path, fold: str):
    with (run / fold / "metrics.csv").open() as stream:
        return list(csv.DictReader(stream))


def pairing_report(run: Path, fold: str, reference: Path | None) -> dict:
    fingerprints = read_json(run / fold / "initial_weight_fingerprints.json")
    rows = metric_rows(run, fold)
    result = {"completed_epochs": len(rows), "initial_state_tensors": len(fingerprints)}
    if reference is None:
        result["reference_pairing"] = "no reference specified for this new baseline"
        return result
    summary = read_json(reference / "summary.json")
    control_run = Path(summary["run"])
    control_fold = summary["fold"]
    control_fingerprints = read_json(control_run / control_fold / "initial_weight_fingerprints.json")
    common = sorted(fingerprints.keys() & control_fingerprints.keys())
    different = [key for key in common if fingerprints[key] != control_fingerprints[key]]
    if not common or different:
        raise RuntimeError(f"Initial shared tensors differ: {different}")
    control_rows = metric_rows(control_run, control_fold)
    target_keys = ["train_paper_target_pixels_1", "train_paper_target_pixels_2",
                   "val_paper_target_pixels_1", "val_paper_target_pixels_2"]
    if len(rows) != len(control_rows):
        raise RuntimeError("Candidate/control epoch budgets differ")
    target_mismatches = [i + 1 for i, (a, b) in enumerate(zip(rows, control_rows, strict=True))
                         if any(float(a[key]) != float(b[key]) for key in target_keys)]
    if target_mismatches:
        raise RuntimeError(f"Per-epoch target totals differ: {target_mismatches}")
    recipe, control_recipe = read_json(run / "resolved_config.json"), read_json(control_run / "resolved_config.json")
    recipe_differences = [name for name in ["data", "train", "loss", "metric", "augmentations", "runtime"]
                          if recipe[name] != control_recipe[name]]
    if recipe_differences:
        raise RuntimeError(f"Matched recipe sections differ: {recipe_differences}")
    result.update(reference_diagnostics=str(reference), original_common_tensors_matched=len(common),
                  original_tensors_absent_from_candidate=sorted(control_fingerprints.keys() - fingerprints.keys()),
                  candidate_additional_tensors=sorted(fingerprints.keys() - control_fingerprints.keys()),
                  per_epoch_target_totals_match=True, target_keys=target_keys,
                  matched_recipe_sections=["data", "train", "loss", "metric", "augmentations", "runtime"])
    return result


def audit_job(root: Path, name: str, job: dict, reference: Path | None, output: Path):
    run, fold = Path(job["run"]), job["fold"]
    diagnostic = root / "diagnostics" / name
    replay_path = root / "verified_experiments" / f"{name}_replay.json"
    replay = read_json(replay_path)
    if replay["validation_samples"] != 444 or not replay["all_required_metrics_match"]:
        raise RuntimeError("Complete clean-fold replay is required")
    summary = read_json(diagnostic / "summary.json")
    if summary["checkpoint_sha256"] != replay["checkpoint_sha256"]:
        raise RuntimeError("Replay and diagnostics use different checkpoints")
    report = {"job": name, "queue": str(root), "run": str(run), "fold": fold,
              "completed_epoch": time.time(), "checkpoint_sha256": replay["checkpoint_sha256"],
              "metric_history_sha256": hashlib.sha256((run / fold / "metrics.csv").read_bytes()).hexdigest(),
              "pairing": pairing_report(run, fold, reference),
              "replay": {"validation_samples": replay["validation_samples"],
                         "all_required_metrics_match": replay["all_required_metrics_match"],
                         "maximum_absolute_difference": max(replay["absolute_differences"].values())}}
    if report["pairing"]["completed_epochs"] != int(job["epochs"]):
        raise RuntimeError("Training budget is incomplete")
    report["optimizer"] = optimizer_audit(run, fold)
    report["instance_detection"] = instance_audit(diagnostic)
    if reference:
        report["image_influence_vs_plain"] = influence_compare(diagnostic, reference)
    report["scope"] = "Additional descriptive evidence; original complete-fold Dice, epoch/threshold selection and migration gates unchanged; no independent test or patient inference"
    atomic_json(output, report)
    return {"status": "complete", "report": str(output),
            "checkpoint_sha256": report["checkpoint_sha256"],
            "completed_epoch": time.time(),
            "shared_macro": summary["reported_macro"]["shared"],
            "instance_f1": report["instance_detection"]["protocols"]["shared"]["pooled_all444"]["f1_all_images"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/overnight_unet_20261012")
    parser.add_argument("--queues", nargs="+", required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    os.chdir(PROJECT_ROOT)
    root = Path(args.root).resolve()
    root.relative_to(PROJECT_ROOT / "outputs")
    window = read_json(root / "window.json")
    evidence = root / "completed_evidence"
    evidence.mkdir(exist_ok=True)
    with (evidence / "observer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = evidence / "state.json"
        state = read_json(state_path) if state_path.exists() else {"jobs": {}}
        state.update(status="running", pid=os.getpid(), process_start=process_start(os.getpid()), queues=args.queues)
        atomic_json(state_path, state)
        while True:
            for queue_name in args.queues:
                queue_root = root / queue_name
                if not (queue_root / "diagnostic_state.json").exists() or not (queue_root / "queue_state.json").exists():
                    continue
                diagnostics, queue = read_json(queue_root / "diagnostic_state.json"), read_json(queue_root / "queue_state.json")
                manifest = read_json(queue_root / "source_manifest.json")
                differences = [name for name, value in manifest["source_sha256"].items()
                               if hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() != value]
                if differences:
                    raise RuntimeError(f"Frozen source changed: {differences}")
                specifications = {row["id"]: row for row in read_json(queue_root / "plan.json")["jobs"]}
                for name, job in queue["jobs"].items():
                    if job["status"] != "complete" or diagnostics["jobs"].get(name, {}).get("status") != "complete":
                        continue
                    if state["jobs"].get(name, {}).get("status") in {"complete", "failed"}:
                        continue
                    recorded_reference = diagnostics["jobs"][name].get("reference")
                    reference_name = specifications[name].get("reference_diagnostics", recorded_reference)
                    reference = Path(reference_name).resolve() if reference_name else None
                    state["jobs"][name] = {"status": "auditing", "started_epoch": time.time()}
                    atomic_json(state_path, state)
                    try:
                        state["jobs"][name] = audit_job(queue_root, name, job, reference, evidence / f"{name}.json")
                        print(f"Full evidence complete: {name}", flush=True)
                    except Exception as error:
                        state["jobs"][name] = {"status": "failed", "error": str(error), "failed_epoch": time.time()}
                        print(f"Evidence failed, inspect before retry: {name}: {error}", flush=True)
                    atomic_json(state_path, state)
            terminal = True
            for queue_name in args.queues:
                qroot = root / queue_name
                queue_path, launch_path = qroot / "queue_state.json", qroot / "launch_status.json"
                if queue_path.exists():
                    q = read_json(queue_path)
                    if q["status"] not in {"complete", "failed"} or any(
                            j["status"] == "complete" and n not in state["jobs"] for n, j in q["jobs"].items()):
                        terminal = False
                elif launch_path.exists():
                    if read_json(launch_path)["status"] not in {"failed", "not_started_insufficient_window", "complete"}:
                        terminal = False
                else:
                    terminal = False
            if args.once or terminal or time.time() >= window["deadline_epoch"]:
                state.update(status="complete" if terminal else "snapshot" if args.once else "window_ended",
                             updated_epoch=time.time())
                atomic_json(state_path, state)
                return
            time.sleep(30)


if __name__ == "__main__":
    main()
