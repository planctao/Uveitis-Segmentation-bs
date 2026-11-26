"""Admit all four additional noise-location arms after complete positive evidence."""

import fcntl
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
from scripts.prepare_noise_skip_replication import screen_evidence
from scripts.run_dino_component_transfer_gate import assert_frozen
from scripts.run_paper_reproduction import atomic_json, process_alive, process_start, run_plan


def read(path):
    return json.loads(path.read_text())


def main():
    os.chdir(PROJECT_ROOT)
    overnight = PROJECT_ROOT / "outputs/overnight_unet_20261012"
    screen, root = overnight / "noise_skip_screen", overnight / "noise_skip_replication"
    root.mkdir(exist_ok=True)
    deadline = read(overnight / "window.json")["deadline_epoch"]
    with (root / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "queue_state.json").exists() or (root / "plan.json").exists():
            raise RuntimeError("Inspect an existing replication plan instead of duplicating admission")
        status_path = root / "launch_status.json"
        status = {"status": "waiting_for_complete_noise_screen", "pid": os.getpid(),
            "process_start": process_start(os.getpid()), "deadline_epoch": deadline,
            "scope": "All four full30epoch arms or no accuracy launch; original fixed bound/location recipe and gate"}
        atomic_json(status_path, status)
        try:
            while True:
                if deadline - time.time() < 3 * 3600:
                    status.update(status="not_started_insufficient_window", minimum_remaining_hours=3.)
                    atomic_json(status_path, status)
                    return
                fallback = read(screen / "launch_status.json")
                if fallback["status"] == "failed":
                    raise RuntimeError("Noise fallback failed; inspect before replication")
                if fallback["status"].startswith("not_started_") or fallback["status"].startswith("probed_not_started_"):
                    status.update(status="not_started_noise_screen_not_admitted", dependency_status=fallback["status"])
                    atomic_json(status_path, status)
                    return
                gate, evidence = screen_evidence(overnight)
                if gate == "not_supported":
                    status.update(status="not_started_original_gate_failed", **{k: v for k, v in evidence.items() if k != "recipes"})
                    atomic_json(status_path, status)
                    return
                if gate == "passed":
                    status.update(**{k: v for k, v in evidence.items() if k != "recipes"})
                    break
                if not process_alive(fallback["pid"], fallback.get("process_start")) and fallback["status"] != "complete":
                    raise RuntimeError("Noise fallback has no live owner")
                time.sleep(20)
            queue = read(screen / "queue_state.json")
            rates = {}
            for name, job in queue["jobs"].items():
                text = (Path(job["run"]) / "f1/train.log").read_text()
                matches = re.findall(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d).*epoch (\d+) started", text, re.M)
                starts = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone(timedelta(hours=8))).timestamp() for t, _ in matches]
                intervals = [b - a for a, b in zip(starts, starts[1:])]
                if len(intervals) < 5:
                    raise RuntimeError("Full-epoch timing evidence missing")
                rates[name] = statistics.median(intervals[-5:])
            required_seconds = max(3 * 3600, sum(2 * (30 * seconds * 1.35 + 120) for seconds in rates.values()))
            status.update(recent5_epoch_seconds_by_arm=rates, required_remaining_seconds=required_seconds,
                          planned_jobs=4, timing_margin_multiplier=1.35, diagnostic_seconds_per_job=120,
                          minimum_free_cuda_mib=2000)
            gpu = read(screen / "real_batch_cuda_optimizer_probe.json")
            if gpu["status"] != "passed" or set(gpu["results"]) != {"skip", "encoder"}:
                raise RuntimeError("Full two-location CUDA probe evidence invalid")
            while True:
                remaining = deadline - time.time()
                if remaining < required_seconds:
                    status.update(status="not_started_insufficient_window", remaining_seconds=remaining)
                    atomic_json(status_path, status)
                    return
                assert_frozen(screen)
                free = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).splitlines()[0])
                if free >= 2000:
                    break
                status.update(status="waiting_for_cuda_memory", free_cuda_mib=free)
                atomic_json(status_path, status)
                time.sleep(20)
            status.update(status="preparing_after_passed_complete_gate")
            atomic_json(status_path, status)
            with (root / "preparation.log").open("a") as stream:
                subprocess.run([sys.executable, "scripts/prepare_noise_skip_replication.py"], stdout=stream, stderr=subprocess.STDOUT, check=True)
            assert_frozen(root)
            if deadline - time.time() < required_seconds:
                status.update(status="prepared_not_started_insufficient_window")
                atomic_json(status_path, status)
                return
            command = [sys.executable, "-u", "scripts/observe_mechanism_screen.py", "--root", str(root)]
            with (root / "observer.log").open("a") as stream:
                observer = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            status.update(status="training", started_epoch=time.time(), observer_pid=observer.pid,
                          observer_start=process_start(observer.pid), original_CUDA_probe_reused=True)
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
