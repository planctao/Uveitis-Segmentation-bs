"""Apply the original CATMIL gate after all full-budget screen arms finish."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.run_paper_reproduction import atomic_json, process_alive, process_start, run_plan


def read(path):
    return json.loads(path.read_text())


def main():
    os.chdir(PROJECT_ROOT)
    overnight = PROJECT_ROOT / "outputs/overnight_unet_20261012"
    screen, root = overnight / "component_screen", overnight / "component_replication"
    root.mkdir(exist_ok=True)
    deadline = read(overnight / "window.json")["deadline_epoch"]
    status_path = root / "launch_status.json"
    with (root / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "queue_state.json").exists() or (root / "plan.json").exists():
            raise RuntimeError("Inspect existing replication plan/queue instead of repeating preparation")
        status = {"status": "waiting_for_complete_screen_and_replay", "pid": os.getpid(),
            "process_start": process_start(os.getpid()), "deadline_epoch": deadline,
            "scope": "Original gate and weights unchanged; complete30epoch runs only"}
        atomic_json(status_path, status)
        try:
            while True:
                if time.time() >= deadline:
                    status.update(status="not_started_window_elapsed")
                    atomic_json(status_path, status)
                    return
                queue, diagnostic = read(screen / "queue_state.json"), read(screen / "diagnostic_state.json")
                if queue["status"] == "failed" or diagnostic["status"] == "failed":
                    raise RuntimeError("Screen/replay failed; inspect before replication")
                if queue["status"] == "complete" and diagnostic["status"] == "complete":
                    break
                if queue["status"] != "complete" and not process_alive(queue["supervisor_pid"], queue.get("supervisor_start")):
                    raise RuntimeError("Initial screen has no live owner")
                time.sleep(20)
            scores = {mode: queue["jobs"][f"catmil_lesion2_{mode}_s42_f1_30_256"]["metrics"]["best_paper_sweep_macro_dice"]
                      for mode in ["cat", "mil", "joint"]}
            scores["plain"] = read(PROJECT_ROOT / "outputs/edg_validation_20261011/diagnostics/paired_concat_s42_f1_30_256/summary.json")["reported_macro"]["shared"]
            candidate = "joint" if all(scores["joint"] > scores[control] for control in ["plain", "cat", "mil"]) else None
            if candidate is None:
                eligible = [mode for mode in ["cat", "mil"] if scores[mode] > scores["plain"]]
                candidate = max(eligible, key=lambda mode: scores[mode]) if eligible else None
            status.update(complete_seed42_scores=scores, candidate=candidate)
            if candidate is None:
                status.update(status="not_started_original_gate_failed")
                atomic_json(status_path, status)
                return
            job_count = 6 if candidate == "joint" else 2
            run = Path(queue["jobs"][f"catmil_lesion2_{candidate}_s42_f1_30_256"]["run"]) / "f1"
            rows = list(csv.DictReader((run / "metrics.csv").open()))
            if len(rows) != 30:
                raise RuntimeError("Initial candidate did not complete30epochs")
            matches = re.findall(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d).*epoch (\d+) started", (run / "train.log").read_text(), re.M)
            starts = [datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone(timedelta(hours=8))).timestamp()
                      for timestamp, _ in matches]
            intervals = [b - a for a, b in zip(starts, starts[1:])]
            if len(intervals) < 5:
                raise RuntimeError("Insufficient full-epoch timing evidence")
            seconds = statistics.median(intervals[-5:])
            required_seconds = job_count * (30 * seconds * 1.35 + 120)
            status.update(planned_jobs=job_count, measured_recent5_epoch_seconds=seconds,
                required_remaining_seconds=required_seconds, timing_margin_multiplier=1.35,
                diagnostic_margin_seconds_per_job=120, minimum_free_cuda_mib=2000)
            while True:
                remaining = deadline - time.time()
                if remaining < required_seconds:
                    status.update(status="not_started_insufficient_full_budget_window", remaining_seconds=remaining)
                    atomic_json(status_path, status)
                    return
                # Query without making a CUDA context in this long-lived supervisor.
                free = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).splitlines()[0])
                if free >= 2000:
                    break
                status.update(status="waiting_for_cuda_memory", free_cuda_mib=free)
                atomic_json(status_path, status)
                time.sleep(20)
            status.update(status="preparing_after_passed_original_gate")
            atomic_json(status_path, status)
            with (root / "preparation.log").open("a") as stream:
                subprocess.run([sys.executable, "scripts/prepare_component_replication.py", "--candidate", candidate],
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
            for name, expected in read(root / "source_manifest.json")["source_sha256"].items():
                if hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f"Frozen replication source changed: {name}")
            if deadline - time.time() < required_seconds:
                status.update(status="prepared_not_started_insufficient_full_budget_window")
                atomic_json(status_path, status)
                return
            command = [sys.executable, "-u", "scripts/observe_mechanism_screen.py", "--root", str(root)]
            with (root / "observer.log").open("a") as stream:
                observer = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            status.update(status="training", started_epoch=time.time(), observer_pid=observer.pid,
                observer_start=process_start(observer.pid), observer_command=command)
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
