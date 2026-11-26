"""Observe completed paired jobs and analyze them without owning their training."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.compare_fusion_diagnostics import paired_comparison
from scripts.run_paper_reproduction import atomic_json, process_alive, process_start
from scripts.verify_paper_experiment import verify_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/edg_validation_20261011")
    args = parser.parse_args()
    root = (PROJECT_ROOT / args.root).resolve()
    root.relative_to(PROJECT_ROOT / "outputs")
    state_path = root / "diagnostic_queue_state.json"
    with (root / "diagnostic_supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(state_path.read_text()) if state_path.exists() else {"jobs": {}}
        state.update(status="running", supervisor_pid=os.getpid(), supervisor_start=process_start(os.getpid()))
        atomic_json(state_path, state)
        plan = json.loads((root / "paired_plan.json").read_text())
        try:
            for job in plan["jobs"]:
                name = job["id"]
                while True:
                    training = json.loads((root / "paired_queue_state.json").read_text())
                    entry = training["jobs"].get(name)
                    if entry and entry.get("status") == "complete":
                        break
                    if training["status"] == "failed" or (entry and entry["status"] == "failed"):
                        raise RuntimeError(f"Training failed for {name}; inspect the training supervisor")
                    if not process_alive(training["supervisor_pid"], training.get("supervisor_start")):
                        child = entry and entry.get("pid")
                        if not child or not process_alive(child, entry.get("process_start")):
                            raise RuntimeError("Training owner is missing; recover it before diagnostics")
                    state.update(waiting_for_training=name)
                    atomic_json(state_path, state)
                    time.sleep(20)
                output = root / "diagnostics" / name
                report = verify_run(Path(entry["run"]), "f1", 30, require_full_data=True)
                previous = state["jobs"].get(name, {})
                if previous.get("status") == "running":
                    while process_alive(previous["pid"], previous.get("process_start")):
                        time.sleep(20)
                summary_path = output / "summary.json"
                accepted = False
                if summary_path.exists():
                    summary = json.loads(summary_path.read_text())
                    accepted = (summary["checkpoint_sha256"] == report["checkpoints"]["best"]["sha256"]
                                and summary["macro_matches_checkpoint"] and summary["validation_samples"] == 444)
                if not accepted:
                    command = [sys.executable, "scripts/analyze_fusion_checkpoint.py", "--run", entry["run"],
                               "--output", str(output), "--save-probabilities"]
                    with (root / f"diagnose_{name}.log").open("a") as stream:
                        child = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT)
                        state["jobs"][name] = {"status": "running", "pid": child.pid,
                                               "process_start": process_start(child.pid), "output": str(output)}
                        atomic_json(state_path, state)
                        code = child.wait()
                    if code != 0:
                        state["jobs"][name].update(status="failed", return_code=code)
                        raise RuntimeError(f"Checkpoint diagnostics failed for {name}")
                    summary = json.loads(summary_path.read_text())
                    if not summary["macro_matches_checkpoint"] or summary["validation_samples"] != 444:
                        raise RuntimeError(f"Diagnostic coverage or replay failed for {name}")
                state["jobs"][name] = {"status": "complete", "output": str(output), "checkpoint_sha256": summary["checkpoint_sha256"]}
                atomic_json(state_path, state)
                print(f"Analyzed {name}", flush=True)
            comparisons = {}
            for seed in plan["seeds"]:
                candidate = root / "diagnostics" / f"paired_edg_s{seed}_f1_30_256"
                for control in ("concat", "equal", "fdg"):
                    directory = root / "diagnostics" / f"paired_{control}_s{seed}_f1_30_256"
                    comparison = paired_comparison(candidate, directory)
                    key = f"s{seed}_edg_minus_{control}"
                    path = root / "diagnostics" / f"{key}.json"
                    atomic_json(path, comparison)
                    comparisons[key] = str(path)
            state.update(status="complete", comparisons=comparisons)
            state.pop("waiting_for_training", None)
            atomic_json(state_path, state)
        except Exception as error:
            state.update(status="failed", error=str(error))
            atomic_json(state_path, state)
            raise


if __name__ == "__main__":
    main()
