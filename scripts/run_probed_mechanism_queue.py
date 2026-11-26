"""Admit a prepared queue after dependency completion and a real CUDA probe."""

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
from scripts.run_paper_reproduction import atomic_json, process_alive, process_start, run_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--dependency-state", required=True)
    parser.add_argument("--window", required=True)
    parser.add_argument("--probe-script", required=True)
    parser.add_argument("--minimum-remaining-hours", type=float, default=4.)
    args = parser.parse_args()
    if args.minimum_remaining_hours < 0:
        parser.error("minimum-remaining-hours must be nonnegative")
    os.chdir(PROJECT_ROOT)
    root = Path(args.root).resolve()
    root.relative_to(PROJECT_ROOT / "outputs")
    dependency = Path(args.dependency_state).resolve()
    probe = Path(args.probe_script).resolve()
    probe.relative_to(PROJECT_ROOT / "scripts")
    window = json.loads(Path(args.window).read_text())
    status_path = root / "launch_status.json"
    with (root / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {"status": "waiting_for_dependency", "pid": os.getpid(),
                  "process_start": process_start(os.getpid()), "dependency": str(dependency),
                  "deadline_epoch": window["deadline_epoch"], "minimum_remaining_hours": args.minimum_remaining_hours}
        atomic_json(status_path, status)
        try:
            while True:
                remaining = float(window["deadline_epoch"]) - time.time()
                if remaining < args.minimum_remaining_hours * 3600:
                    status.update(status="not_started_insufficient_window", remaining_seconds=remaining)
                    atomic_json(status_path, status)
                    return
                if dependency.exists():
                    state = json.loads(dependency.read_text())
                    if state["status"] == "complete":
                        break
                    if state["status"] == "failed":
                        raise RuntimeError("Dependency failed; inspect before admitting more GPU work")
                    owner_live = process_alive(state["supervisor_pid"], state.get("supervisor_start"))
                    child_live = any(j.get("status") == "running" and process_alive(j["pid"], j.get("process_start"))
                                     for j in state["jobs"].values())
                    if not owner_live and not child_live:
                        raise RuntimeError("Dependency has no live owner or child")
                time.sleep(20)
            manifest = json.loads((root / "source_manifest.json").read_text())
            for name, expected in manifest["source_sha256"].items():
                if hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f"Frozen training source changed: {name}")
            command = [sys.executable, str(probe), "--device", "cuda", "--optimizer-memory"]
            while True:
                remaining = float(window["deadline_epoch"]) - time.time()
                if remaining < args.minimum_remaining_hours * 3600:
                    status.update(status="not_started_insufficient_window", remaining_seconds=remaining)
                    atomic_json(status_path, status)
                    return
                status.update(status="checking_cuda_memory", probe_command=command)
                atomic_json(status_path, status)
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                with (root / "admission_probe.log").open("a") as stream:
                    stream.write(result.stdout)
                if result.returncode == 0:
                    status.update(probe_passed=True, probe_completed_epoch=time.time())
                    break
                if "Defer whole-model probe:" not in result.stdout:
                    raise RuntimeError("Full CUDA probe failed; see admission_probe.log; no accuracy job started")
                status.update(status="waiting_for_cuda_memory")
                atomic_json(status_path, status)
                time.sleep(20)
            remaining = float(window["deadline_epoch"]) - time.time()
            if remaining < args.minimum_remaining_hours * 3600:
                status.update(status="not_started_insufficient_window", remaining_seconds=remaining)
                atomic_json(status_path, status)
                return
            for name, expected in manifest["source_sha256"].items():
                if hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f"Training source changed during admission: {name}")
            status.update(status="training", started_epoch=time.time())
            atomic_json(status_path, status)
            run_plan(root / "plan.json", root / "queue_state.json", resume=True)
            status.update(status="complete", completed_epoch=time.time())
            atomic_json(status_path, status)
        except Exception as error:
            status.update(status="failed", error=str(error))
            atomic_json(status_path, status)
            raise


if __name__ == "__main__":
    main()
