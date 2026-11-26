"""Run the planned f1 controls after full ConDSeg releases its GPU workload.

These jobs have the exact specifications used by promotion. Promotion observes
this worker through the existing external-job ownership mechanism; it never
launches a second copy of either control.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.promote_paper_reproduction import control_jobs
from scripts.run_paper_reproduction import PROJECT_ROOT, atomic_json, process_alive, process_start, run_plan
from scripts.verify_paper_experiment import verify_run


def wait_for_full_condseg(root: Path) -> dict:
    while True:
        screen = json.loads((root / "screen_state.json").read_text())
        job = screen["jobs"].get("author_condseg_f1_30_256")
        if job and job["status"] == "complete":
            if job.get("return_code") != 0:
                raise RuntimeError("Full ConDSeg has no captured successful return code")
            report = verify_run(Path(job["run"]), "f1", 30, require_full_data=True)
            if report["metrics"] != job["metrics"]:
                raise RuntimeError("Full ConDSeg queue metrics differ from verified checkpoint")
            return report
        if screen["status"] == "failed" or (job and job["status"] == "failed"):
            raise RuntimeError("Screen failed before full ConDSeg completed; diagnose it first")
        supervisor_live = process_alive(screen["supervisor_pid"], screen.get("supervisor_start"))
        child_live = bool(job and process_alive(job["pid"], job.get("process_start")))
        if not supervisor_live and not child_live:
            raise RuntimeError("Full ConDSeg worker and child are missing")
        print("waiting for the existing full ConDSeg training to finish", flush=True)
        time.sleep(20)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/paper_reproduction_20261011")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    torch.set_num_threads(2)
    root = Path(args.root).resolve()
    plan_path, state_path = root / "control_precompute_plan.json", root / "control_precompute_state.json"
    root.mkdir(parents=True, exist_ok=True)
    jobs = control_jobs(["f1"], 256) + control_jobs(["f1"], 768)
    with state_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if plan_path.exists():
            if json.loads(plan_path.read_text())["jobs"] != jobs:
                raise RuntimeError("Existing control plan differs from the promotion specifications")
        else:
            atomic_json(plan_path, {"description": "Exact planned f1 controls, owned independently after full ConDSeg", "jobs": jobs})
        state = json.loads(state_path.read_text()) if state_path.exists() else {"jobs": {}}
        state.update(status="waiting_for_condseg", supervisor_pid=os.getpid(), supervisor_start=process_start(os.getpid()),
                     plan=str(plan_path), plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest())
        atomic_json(state_path, state)
        try:
            dependency = wait_for_full_condseg(root)
            atomic_json(root / "control_precompute_dependency_audit.json", dependency)
            run_plan(plan_path, state_path, args.resume)
        except Exception as error:
            state = json.loads(state_path.read_text())
            atomic_json(state_path, {**state, "status": "failed", "error": str(error)})
            raise


if __name__ == "__main__":
    main()
