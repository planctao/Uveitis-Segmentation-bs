"""Run the newly researched U-Net papers after the existing promotion queue.

The current matched-resolution supervisor owns the GPU.  This process only
observes its authoritative promotion status and launches the new paper queue
after that queue reaches ``complete``.  It never starts a duplicate while the
existing supervisor is live, and it records a terminal failure if the existing
queue fails or disappears.
"""

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

from scripts.run_paper_reproduction import atomic_json, process_alive, run_plan  # noqa: E402
from scripts.verify_paper_experiment import verify_run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/paper_reproduction_20261011")
    parser.add_argument("--plan", default="outputs/paper_reproduction_20261011/recent_priority_plan.json")
    parser.add_argument("--wait-status", default="outputs/paper_reproduction_20261011/promotion_status.json")
    parser.add_argument("--state", default="outputs/paper_reproduction_20261011/recent_priority_state.json")
    parser.add_argument(
        "--start-immediately",
        action="store_true",
        help="Start the recent-paper queue while the older promotion queue is still running.",
    )
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    root = Path(args.root).resolve()
    plan = Path(args.plan).resolve()
    status_path = Path(args.wait_status).resolve()
    state_path = Path(args.state).resolve()
    lock_path = root / "recent_priority_queue.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = json.loads(state_path.read_text()) if state_path.exists() else {}
        if existing.get("status") == "complete":
            return
        atomic_json(state_path, {**existing, "status": "waiting_for_existing_promotion", "supervisor_pid": os.getpid()})
        if args.start_immediately:
            # The recent-paper models are the active objective.  They have
            # separate output/state files, so they can proceed while the
            # older matched-resolution queue keeps its own GPU process alive.
            current = json.loads(status_path.read_text()) if status_path.exists() else {}
            atomic_json(
                state_path,
                {
                    "status": "running",
                    "jobs": existing.get("jobs", {}),
                    "supervisor_pid": os.getpid(),
                    "plan": str(plan),
                    "started_while_existing_promotion": True,
                    "existing_promotion_status": current,
                },
            )
        else:
            while True:
                if not status_path.exists():
                    time.sleep(30)
                    continue
                current = json.loads(status_path.read_text())
                status = current.get("status")
                if status == "complete":
                    break
                if status in {"failed", "screen_failed_requires_diagnosis"}:
                    atomic_json(state_path, {"status": "blocked_by_existing_failure", "existing_status": current})
                    raise RuntimeError(f"Existing paper promotion queue failed: {current}")
                supervisor = current.get("supervisor_pid")
                if supervisor:
                    if current.get("supervisor_start"):
                        live = process_alive(int(supervisor), current.get("supervisor_start"))
                    else:
                        # Older promotion status records did not persist the
                        # start token; /proc existence is still safer than
                        # launching a second GPU queue on an active PID.
                        live = Path(f"/proc/{int(supervisor)}").exists()
                else:
                    live = False
                if not live:
                    atomic_json(state_path, {"status": "blocked_by_missing_existing_supervisor", "existing_status": current})
                    raise RuntimeError(f"Existing promotion supervisor disappeared: {current}")
                time.sleep(60)
            atomic_json(
                state_path,
                {
                    "status": "running",
                    "jobs": existing.get("jobs", {}),
                    "supervisor_pid": os.getpid(),
                    "plan": str(plan),
                },
            )
        try:
            run_plan(plan, state_path, resume=True)
            final = json.loads(state_path.read_text())
            if final.get("status") != "complete":
                atomic_json(state_path, {**final, "status": "failed", "error": "run_plan returned without complete state"})
                raise RuntimeError("Recent paper queue did not reach complete state")
            atomic_json(state_path, {**final, "status": "auditing_checkpoints", "supervisor_pid": os.getpid()})
            audit_dir = root / "verified_experiments"
            audit_dir.mkdir(parents=True, exist_ok=True)
            audit_reports = {}
            for job in json.loads(plan.read_text())["jobs"]:
                name, fold = job["id"], job.get("fold", "f1")
                entry = final["jobs"][name]
                report = verify_run(Path(entry["run"]), fold, int(entry["epochs"]), require_full_data=True)
                audit_path = audit_dir / f"{name}.json"
                atomic_json(audit_path, {"phase": "recent_paper", **report})
                replay_path = audit_dir / f"{name}_full_checkpoint_evaluation.json"
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/audit_paper_checkpoint_evaluation.py",
                        "--run",
                        str(entry["run"]),
                        "--fold",
                        fold,
                        "--output",
                        str(replay_path),
                    ],
                    cwd=PROJECT_ROOT,
                    check=True,
                )
                replay = json.loads(replay_path.read_text())
                if not replay.get("all_required_metrics_match") or int(replay.get("validation_samples", 0)) != 444:
                    raise RuntimeError(f"Full clean-fold replay failed for {name}: {replay}")
                audit_reports[f"{name}_full_checkpoint_evaluation"] = str(replay_path)
                audit_reports[name] = str(audit_path)
            audited = json.loads(state_path.read_text())
            atomic_json(state_path, {**audited, "status": "complete", "audit_reports": audit_reports})
        except Exception as error:
            previous = json.loads(state_path.read_text()) if state_path.exists() else {}
            atomic_json(state_path, {**previous, "status": "failed", "error": str(error)})
            raise


if __name__ == "__main__":
    main()
