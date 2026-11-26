"""Schedule a new GT-region direction without changing the original BiCC gate."""

import argparse
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
from scripts.prepare_cc_ground_truth_followup import screen_evidence
from scripts.run_dino_component_transfer_gate import assert_frozen
from scripts.run_paper_reproduction import atomic_json, process_alive, process_start, run_plan


def read(path):
    return json.loads(path.read_text())


def observe_evidence(root, deadline):
    """Disjoint job reports and a local state; never rewrite the main observer state."""
    import torch
    from scripts.audit_completed_mechanism_evidence import audit_job
    torch.set_num_threads(2)
    output = root.parent / "completed_evidence"
    state_path = root / "followup_evidence_state.json"
    state = read(state_path) if state_path.exists() else {"jobs": {}}
    state.update(status="running", pid=os.getpid(), process_start=process_start(os.getpid()))
    with (root / "followup_evidence.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            assert_frozen(root)
            atomic_json(state_path, state)
            if (root / "queue_state.json").exists() and (root / "diagnostic_state.json").exists():
                queue, diagnostic = read(root / "queue_state.json"), read(root / "diagnostic_state.json")
                specifications = {j["id"]: j for j in read(root / "plan.json")["jobs"]}
                for name, job in queue["jobs"].items():
                    if job["status"] != "complete" or diagnostic["jobs"].get(name, {}).get("status") != "complete":
                        continue
                    if state["jobs"].get(name, {}).get("status") in {"complete", "failed"}:
                        continue
                    state["jobs"][name] = {"status": "auditing", "started_epoch": time.time()}
                    atomic_json(state_path, state)
                    try:
                        reference = PROJECT_ROOT / specifications[name]["reference_diagnostics"]
                        state["jobs"][name] = audit_job(root, name, job, reference, output / f"{name}.json")
                    except Exception as error:
                        state["jobs"][name] = {"status": "failed", "error": str(error)}
                    atomic_json(state_path, state)
                if queue["status"] == "complete" and len(state["jobs"]) == len(specifications) and all(
                        v["status"] in {"complete", "failed"} for v in state["jobs"].values()):
                    state.update(status="complete" if all(v["status"] == "complete" for v in state["jobs"].values()) else "failed")
                    atomic_json(state_path, state)
                    return
            if time.time() >= deadline:
                state.update(status="window_ended")
                atomic_json(state_path, state)
                return
            time.sleep(20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observe-evidence", action="store_true")
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    overnight = PROJECT_ROOT / "outputs/overnight_unet_20261012"
    root = overnight / "cc_gt_followup"
    root.mkdir(exist_ok=True)
    deadline = read(overnight / "window.json")["deadline_epoch"]
    if args.observe_evidence:
        observe_evidence(root, deadline)
        return
    with (root / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "plan.json").exists() or (root / "queue_state.json").exists():
            raise RuntimeError("Inspect existing follow-up instead of duplicating")
        status_path = root / "launch_status.json"
        status = {"status": "waiting_for_complete_BiCC_screen", "pid": os.getpid(),
                  "process_start": process_start(os.getpid()), "deadline_epoch": deadline,
                  "scope": "New GT-region direction, global-only seed42 plus GT-only seeds2026/3407; all full30epoch tasks or no admission"}
        atomic_json(status_path, status)
        try:
            while True:
                if deadline - time.time() < 3 * 3600:
                    status.update(status="not_started_insufficient_window", minimum_remaining_hours=3.)
                    atomic_json(status_path, status)
                    return
                gate, evidence = screen_evidence(overnight)
                if gate == "not_supported":
                    status.update(status="not_started_independent_GT_gate_failed", **{k: v for k, v in evidence.items() if k != "recipe"})
                    atomic_json(status_path, status)
                    return
                if gate == "passed":
                    status.update(**{k: v for k, v in evidence.items() if k != "recipe"})
                    break
                queue = read(overnight / "bicc_screen/queue_state.json")
                if queue["status"] != "complete" and not process_alive(queue["supervisor_pid"], queue.get("supervisor_start")):
                    raise RuntimeError("Original BiCC queue has no live owner")
                time.sleep(20)
            queue = read(overnight / "bicc_screen/queue_state.json")
            rates = {}
            for name, job in queue["jobs"].items():
                log = (Path(job["run"]) / "f1/train.log").read_text()
                matches = re.findall(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d).*epoch (\d+) started", log, re.M)
                starts = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone(timedelta(hours=8))).timestamp() for t, _ in matches]
                intervals = [b - a for a, b in zip(starts, starts[1:])]
                if len(intervals) < 5:
                    raise RuntimeError("Recent full-epoch timing evidence missing")
                rates[name] = statistics.median(intervals[-5:])
            seconds = max(rates.values())
            required = max(3 * 3600, 3 * (30 * seconds * 1.35 + 120) + 180)
            status.update(recent5_epoch_seconds_by_arm=rates, conservative_epoch_seconds=seconds,
                          required_remaining_seconds=required, planned_jobs=3, timing_margin_multiplier=1.35,
                          minimum_free_cuda_mib=3200, existing_trainers_not_preempted=True)
            if deadline - time.time() < required:
                status.update(status="not_started_insufficient_window", remaining_seconds=deadline - time.time())
                atomic_json(status_path, status)
                return
            status.update(status="preparing_independent_followup")
            atomic_json(status_path, status)
            with (root / "preparation.log").open("a") as stream:
                subprocess.run([sys.executable, "scripts/prepare_cc_ground_truth_followup.py"], stdout=stream, stderr=subprocess.STDOUT, check=True)
            while True:
                if deadline - time.time() < required:
                    status.update(status="prepared_not_started_insufficient_window", remaining_seconds=deadline - time.time())
                    atomic_json(status_path, status)
                    return
                assert_frozen(root)
                command = [sys.executable, "scripts/probe_cc_ground_truth_followup.py", "--device", "cuda", "--optimizer-memory"]
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                with (root / "admission_probe.log").open("a") as stream:
                    stream.write(result.stdout)
                if result.returncode == 0:
                    break
                if "Defer whole-model probe:" not in result.stdout:
                    raise RuntimeError("Follow-up full CUDA probe failed; no accuracy training launched")
                status.update(status="waiting_for_cuda_memory")
                atomic_json(status_path, status)
                time.sleep(20)
            if deadline - time.time() < required:
                status.update(status="probed_not_started_insufficient_window")
                atomic_json(status_path, status)
                return
            assert_frozen(root)
            observers = {}
            for label, command in [
                ("diagnostics", [sys.executable, "-u", "scripts/observe_mechanism_screen.py", "--root", str(root)]),
                ("evidence", [sys.executable, "-u", "scripts/run_cc_ground_truth_followup.py", "--observe-evidence"])]:
                with (root / f"{label}_observer.log").open("a") as stream:
                    child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                observers[label] = {"pid": child.pid, "process_start": process_start(child.pid)}
            status.update(status="training", started_epoch=time.time(), observers=observers, full_CUDA_probe_passed=True)
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
