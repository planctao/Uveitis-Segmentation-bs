"""Verify and diagnose completed jobs while a mechanism queue continues."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.run_paper_reproduction import atomic_json, process_alive, process_start
from scripts.verify_paper_experiment import verify_run
from scripts.compare_fusion_diagnostics import paired_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--reference", help="Existing paired baseline diagnostic directory; omit for a new baseline audit")
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    root = Path(args.root).resolve()
    root.relative_to(PROJECT_ROOT / "outputs")
    reference = Path(args.reference).resolve() if args.reference else None
    plan = json.loads((root / "plan.json").read_text())
    specifications = {job["id"]: job for job in plan["jobs"]}
    state_path = root / "diagnostic_state.json"
    with (root / "diagnostic.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(state_path.read_text()) if state_path.exists() else {"jobs": {}}
        if state.get("status") == "complete":
            return
        state.update(status="running", pid=os.getpid(), process_start=process_start(os.getpid()))
        atomic_json(state_path, state)
        try:
            while True:
                queue_path = root / "queue_state.json"
                if not queue_path.exists():
                    time.sleep(10)
                    continue
                queue = json.loads(queue_path.read_text())
                for name, job in queue["jobs"].items():
                    if job["status"] != "complete" or state["jobs"].get(name, {}).get("status") == "complete":
                        continue
                    state["jobs"][name] = {"status": "auditing", "run": job["run"]}
                    atomic_json(state_path, state)
                    run = Path(job["run"])
                    verified = verify_run(run, job["fold"], int(job["epochs"]), require_full_data=True)
                    atomic_json(root / "verified_experiments" / f"{name}.json", verified)
                    diagnostic = root / "diagnostics" / name
                    commands = [
                        [sys.executable, "scripts/analyze_fusion_checkpoint.py", "--run", str(run),
                         "--output", str(diagnostic), "--save-probabilities"],
                        [sys.executable, "scripts/audit_paper_checkpoint_evaluation.py", "--run", str(run),
                         "--output", str(root / "verified_experiments" / f"{name}_replay.json")],
                    ]
                    with (root / f"diagnostic_{name}.log").open("a") as stream:
                        for command in commands:
                            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
                    reference_name = specifications[name].get("reference_diagnostics", str(reference) if reference else None)
                    job_reference = Path(reference_name).resolve() if reference_name else None
                    comparison = paired_comparison(diagnostic, job_reference) if job_reference else None
                    if comparison is not None:
                        atomic_json(root / "comparisons" / f"{name}_vs_reference.json", comparison)
                    state["jobs"][name].update(status="complete", metrics=job["metrics"],
                                               reference=str(job_reference) if job_reference else None)
                    if comparison is not None:
                        state["jobs"][name]["shared_macro_delta"] = comparison["comparisons"]["shared"]["macro_delta"]
                    atomic_json(state_path, state)
                    print(f"Verified and diagnosed {name}", flush=True)
                if queue["status"] == "complete":
                    if set(state["jobs"]) != set(queue["jobs"]) or any(j["status"] != "complete" for j in state["jobs"].values()):
                        raise RuntimeError("Completed training queue has incomplete diagnostics")
                    initial = {name: json.loads((Path(job["run"]) / job["fold"] / "initial_weight_fingerprints.json").read_text())
                               for name, job in queue["jobs"].items()}
                    first = next(iter(initial.values()))
                    identical = all(value == first for value in initial.values())
                    report = {"jobs": state["jobs"], "initial_states_identical_across_arms": identical,
                              "reference": str(reference) if reference else None, "paired_comparisons": {},
                              "scope": "exploratory mechanism screen on the same validation fold used for epoch and threshold selection"}
                    names = list(queue["jobs"])
                    report["initialization_by_group"] = {}
                    groups = {name: str(specifications[name].get("comparison_group", "default")) for name in names}
                    for group in sorted(set(groups.values())):
                        members = [name for name in names if groups[name] == group]
                        reference_state = initial[members[0]]
                        differences = {name: [key for key, value in initial[name].items()
                                              if reference_state.get(key) != value] for name in members}
                        allowed = plan.get("expected_initial_difference_suffixes", [])
                        unexpected = {name: [key for key in keys if not any(key.endswith(suffix) for suffix in allowed)]
                                      for name, keys in differences.items()}
                        if any(unexpected.values()):
                            raise RuntimeError(f"Unexpected initial tensor differences in {group}: {unexpected}")
                        report["initialization_by_group"][group] = {"members": members,
                            "differing_fixed_tensors": differences, "only_expected_differences": True}
                    for i, name in enumerate(names):
                        for other in names[i + 1:]:
                            if groups[name] != groups[other]:
                                continue
                            report["paired_comparisons"][f"{name}_vs_{other}"] = paired_comparison(
                                root / "diagnostics" / name, root / "diagnostics" / other)
                    report["metric_history_sha256"] = {
                        name: hashlib.sha256((Path(job["run"]) / job["fold"] / "metrics.csv").read_bytes()).hexdigest()
                        for name, job in queue["jobs"].items()}
                    atomic_json(root / "screen_report.json", report)
                    state.update(status="complete", completed_epoch=time.time(), report=str(root / "screen_report.json"))
                    atomic_json(state_path, state)
                    return
                if queue["status"] == "failed":
                    raise RuntimeError("Training queue failed; inspect its recorded logs before recovery")
                if not process_alive(queue["supervisor_pid"], queue.get("supervisor_start")):
                    live = any(j.get("status") == "running" and process_alive(j["pid"], j.get("process_start"))
                               for j in queue["jobs"].values())
                    if not live:
                        raise RuntimeError("Training queue has no live owner or child; recover its recorded state")
                time.sleep(20)
        except Exception as error:
            state.update(status="failed", error=str(error))
            atomic_json(state_path, state)
            raise


if __name__ == "__main__":
    main()
