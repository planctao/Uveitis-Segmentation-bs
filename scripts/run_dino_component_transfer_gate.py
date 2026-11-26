"""Admit DINO512 CAT only after full, positive three-seed mechanism evidence."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.run_paper_reproduction import atomic_json, process_alive, process_start, run_plan


def read(path):
    return json.loads(path.read_text())


def assert_frozen(root):
    for name, expected in read(root / "source_manifest.json")["source_sha256"].items():
        if hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen source changed: {name}")


def gate_evidence(overnight):
    replication = overnight / "component_replication"
    if not (replication / "queue_state.json").exists():
        launch = read(replication / "launch_status.json")
        if launch["status"] in {"failed", "not_started_original_gate_failed", "not_started_insufficient_full_budget_window",
                                "not_started_window_elapsed", "prepared_not_started_insufficient_full_budget_window"}:
            return "not_supported", {"replication_launch_status": launch["status"]}
        return "pending", {}
    plan = read(replication / "plan.json")
    if plan["candidate"] != "cat":
        return "not_supported", {"reason": "Prepared strong CAT requires CAT replication; joint requires additional strong single-term controls"}
    queue, diagnostic = read(replication / "queue_state.json"), read(replication / "diagnostic_state.json")
    if queue["status"] == "failed" or diagnostic["status"] == "failed":
        raise RuntimeError("Replication training/replay failed")
    if queue["status"] != "complete" or diagnostic["status"] != "complete":
        return "pending", {}
    scores, reports = {}, {}
    for seed in [42, 2026, 3407]:
        name = f"catmil_lesion2_cat_s{seed}_f1_30_256"
        evidence_path = overnight / "completed_evidence" / f"{name}.json"
        if not evidence_path.exists():
            return "pending", {"waiting_for_extra_completed_evidence": name}
        report = read(evidence_path)
        if report["pairing"]["completed_epochs"] != 30 or report["pairing"]["original_common_tensors_matched"] != 292:
            raise RuntimeError("CAT full-budget original initialization evidence invalid")
        if not report["pairing"]["per_epoch_target_totals_match"] or not report["replay"]["all_required_metrics_match"]:
            raise RuntimeError("CAT original pairing/replay evidence invalid")
        queue_root = overnight / ("component_screen" if seed == 42 else "component_replication")
        summary = read(queue_root / "diagnostics" / name / "summary.json")
        reference = read(PROJECT_ROOT / f"outputs/edg_validation_20261011/diagnostics/paired_concat_s{seed}_f1_30_256/summary.json")
        scores[str(seed)] = {"cat": summary["reported_macro"]["shared"], "plain": reference["reported_macro"]["shared"],
                            "delta": summary["reported_macro"]["shared"] - reference["reported_macro"]["shared"]}
        reports[str(seed)] = str(evidence_path)
    return ("passed" if all(value["delta"] > 0 for value in scores.values()) else "not_supported"), {
        "complete_three_seed_scores": scores, "evidence_reports": reports,
        "scope": "Same model-selection fold and original positive-all-three-seeds gate; no independent test claim"}


def defer_waiting_prototype(overnight, status):
    """Reserve released resources for a fully supported higher-priority candidate."""
    prototype = overnight / "prototype_screen"
    launch_path = prototype / "launch_status.json"
    launch = read(launch_path)
    if (prototype / "queue_state.json").exists() or launch["status"] != "waiting_for_dependency":
        return
    pid, start = launch["pid"], launch.get("process_start")
    if not process_alive(pid, start):
        return
    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    if b"run_probed_mechanism_queue.py" not in cmdline or b"prototype_screen" not in cmdline:
        raise RuntimeError("Waiting PDR supervisor identity mismatch")
    # Freeze the waiting supervisor while checking its children and queue. A
    # dependency may finish between the first read and signal delivery. Never
    # terminate an owner that has already started a probe or accuracy worker.
    os.kill(pid, signal.SIGSTOP)
    deferred = False
    try:
        stopped = False
        for _ in range(25):
            if not process_alive(pid, start):
                return
            process_state = Path(f"/proc/{pid}/status").read_text()
            if any(line.startswith("State:") and "T (stopped)" in line for line in process_state.splitlines()):
                stopped = True
                break
            time.sleep(.02)
        if not stopped:
            raise RuntimeError("Could not freeze the waiting PDR owner for a race-free child check")
        children = Path(f"/proc/{pid}/task/{pid}/children").read_text().strip()
        current = read(launch_path)
        if children or (prototype / "queue_state.json").exists() or current["status"] != "waiting_for_dependency":
            return
        atomic_json(prototype / "launch_status_before_supported_CAT_priority.json", current)
        os.kill(pid, signal.SIGTERM)
        deferred = True
    finally:
        if process_alive(pid, start):
            os.kill(pid, signal.SIGCONT)
    if not deferred:
        return
    launch.update(status="deferred_for_replicated_DINO512_CAT", deferred_epoch=time.time(),
                  reason="Complete positive three-seed CAT evidence; reserve released GPU memory for matched strong baseline objective candidate",
                  training_recipe_changed=False, existing_training_stopped=False)
    atomic_json(launch_path, launch)
    status.update(waiting_PDR_supervisor_deferred=pid, prototype_accuracy_training_started=False)


def main():
    os.chdir(PROJECT_ROOT)
    overnight = PROJECT_ROOT / "outputs/overnight_unet_20261012"
    root = overnight / "dino_component_transfer_512"
    deadline = read(overnight / "window.json")["deadline_epoch"]
    status_path = root / "launch_status.json"
    with (root / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "queue_state.json").exists():
            raise RuntimeError("Inspect an existing strong queue instead of repeating admission")
        if read(root / "plan.json")["mode"] != "cat":
            raise ValueError("This gate admits only the independently replicated CAT objective")
        status = {"status": "waiting_for_complete_three_seed_evidence", "pid": os.getpid(),
            "process_start": process_start(os.getpid()), "deadline_epoch": deadline,
            "minimum_remaining_hours": 4., "minimum_free_cuda_mib": 6800,
            "timing_basis": "Baseline last5full epochs median435s: full30epochs3.625h plus22.5minute overhead/diagnostic allowance; contention remains uncertain"}
        atomic_json(status_path, status)
        try:
            while True:
                remaining = deadline - time.time()
                if remaining < 4 * 3600:
                    status.update(status="not_started_insufficient_full30epoch_window", remaining_seconds=remaining)
                    atomic_json(status_path, status)
                    return
                gate, evidence = gate_evidence(overnight)
                if gate == "not_supported":
                    status.update(status="not_started_original_replication_gate_failed", **evidence)
                    atomic_json(status_path, status)
                    return
                if gate == "passed":
                    status.update(three_seed_gate_passed=True, **evidence)
                    break
                time.sleep(20)
            assert_frozen(root)
            cpu_report = read(root / "real_batch_cpu_optimizer_probe.json")
            if cpu_report["status"] != "passed" or cpu_report["initial_original_state_tensors_matched"] != 207:
                raise RuntimeError("CPU whole-model/initialization evidence invalid")
            defer_waiting_prototype(overnight, status)
            while True:
                remaining = deadline - time.time()
                if remaining < 4 * 3600:
                    status.update(status="not_started_insufficient_full30epoch_window", remaining_seconds=remaining)
                    atomic_json(status_path, status)
                    return
                free = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).splitlines()[0])
                status.update(status="waiting_for_full_CUDA_admission", remaining_seconds=remaining, free_cuda_mib=free)
                atomic_json(status_path, status)
                if free < 6800:
                    time.sleep(20)
                    continue
                command = [sys.executable, "scripts/probe_dino_component_real_training.py", "--device", "cuda", "--optimizer-memory"]
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                with (root / "admission_probe.log").open("a") as stream:
                    stream.write(result.stdout)
                if result.returncode == 0:
                    break
                if "Defer whole-model probe:" not in result.stdout:
                    raise RuntimeError("Full CUDA probe failed; no accuracy training started")
                time.sleep(20)
            gpu_report = read(root / "real_batch_cuda_optimizer_probe.json")
            if gpu_report["status"] != "passed" or gpu_report["optimizer_steps"] != 1:
                raise RuntimeError("Full CUDA optimizer-memory probe did not pass")
            for field in ["images_sha256", "masks_sha256", "sample_ids", "folds", "images_shape"]:
                if gpu_report[field] != cpu_report[field]:
                    raise RuntimeError(f"CUDA/CPU paired training batch differs: {field}")
            assert_frozen(root)
            if deadline - time.time() < 4 * 3600:
                status.update(status="probed_not_started_insufficient_full30epoch_window")
                atomic_json(status_path, status)
                return
            command = [sys.executable, "-u", "scripts/observe_mechanism_screen.py", "--root", str(root)]
            with (root / "observer.log").open("a") as stream:
                observer = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            status.update(status="training", cuda_full_optimizer_probe_passed=True, started_epoch=time.time(),
                observer_pid=observer.pid, observer_start=process_start(observer.pid), observer_command=command)
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
