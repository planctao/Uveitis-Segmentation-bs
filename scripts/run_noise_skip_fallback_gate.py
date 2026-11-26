"""Run fixed NoiseUNet controls only after complete CAT replication rejects transfer."""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.run_dino_component_transfer_gate import assert_frozen, gate_evidence
from scripts.run_paper_reproduction import atomic_json, process_start, run_plan


def read(path):
    return json.loads(path.read_text())


def main():
    os.chdir(PROJECT_ROOT)
    overnight = PROJECT_ROOT / "outputs/overnight_unet_20261012"
    root, replication = overnight / "noise_skip_screen", overnight / "component_replication"
    deadline = read(overnight / "window.json")["deadline_epoch"]
    with (root / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "queue_state.json").exists():
            raise RuntimeError("Inspect existing fallback instead of duplicating it")
        status_path = root / "launch_status.json"
        status = {"status": "waiting_for_complete_CAT_replication_decision", "pid": os.getpid(),
            "process_start": process_start(os.getpid()), "deadline_epoch": deadline,
            "minimum_remaining_hours": 3., "scope": "Lower-priority two full30epoch location arms; no CAT/MIL stacking"}
        atomic_json(status_path, status)
        try:
            while True:
                remaining = deadline - time.time()
                if remaining < 3 * 3600:
                    status.update(status="not_started_insufficient_full_two_arm_window", remaining_seconds=remaining)
                    atomic_json(status_path, status)
                    return
                if read(replication / "launch_status.json")["status"] == "failed":
                    raise RuntimeError("Replication admission failed; inspect before more work")
                if (replication / "queue_state.json").exists():
                    queue = read(replication / "queue_state.json")
                    diagnostic_path = replication / "diagnostic_state.json"
                    if queue["status"] == "failed":
                        raise RuntimeError("CAT replication failed; inspect before more work")
                    if diagnostic_path.exists() and read(diagnostic_path)["status"] == "failed":
                        raise RuntimeError("CAT full replay failed; inspect before more work")
                    if queue["status"] != "complete" or not diagnostic_path.exists() or read(diagnostic_path)["status"] != "complete":
                        time.sleep(20)
                        continue
                gate, evidence = gate_evidence(overnight)
                if gate == "passed":
                    status.update(status="not_started_supported_CAT_has_priority", CAT_gate=evidence)
                    atomic_json(status_path, status)
                    return
                if gate == "not_supported":
                    status.update(CAT_complete_decision=evidence)
                    break
                time.sleep(20)
            cpu = read(root / "real_batch_cpu_optimizer_probe.json")
            if cpu["status"] != "passed" or any(value["initial_original_tensors_matched"] != 292 for value in cpu["results"].values()):
                raise RuntimeError("CPU whole-model initialization probe invalid")
            while True:
                remaining = deadline - time.time()
                if remaining < 3 * 3600:
                    status.update(status="not_started_insufficient_full_two_arm_window", remaining_seconds=remaining)
                    atomic_json(status_path, status)
                    return
                assert_frozen(root)
                status.update(status="checking_full_CUDA_admission", remaining_seconds=remaining)
                atomic_json(status_path, status)
                command = [sys.executable, "scripts/probe_noise_skip_real_training.py", "--device", "cuda", "--optimizer-memory"]
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                with (root / "admission_probe.log").open("a") as stream:
                    stream.write(result.stdout)
                if result.returncode == 0:
                    break
                if "Defer whole-model probe:" not in result.stdout:
                    raise RuntimeError("CUDA probe failed; no accuracy training started")
                time.sleep(20)
            gpu = read(root / "real_batch_cuda_optimizer_probe.json")
            for field in ["images_sha256", "masks_sha256", "sample_ids", "folds", "images_shape"]:
                if gpu[field] != cpu[field]:
                    raise RuntimeError(f"CPU/CUDA paired batch differs: {field}")
            assert_frozen(root)
            if deadline - time.time() < 3 * 3600:
                status.update(status="probed_not_started_insufficient_full_two_arm_window")
                atomic_json(status_path, status)
                return
            command = [sys.executable, "-u", "scripts/observe_mechanism_screen.py", "--root", str(root)]
            with (root / "observer.log").open("a") as stream:
                observer = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            status.update(status="training", started_epoch=time.time(), CUDA_full_optimizer_probe_passed=True,
                          observer_pid=observer.pid, observer_start=process_start(observer.pid))
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
