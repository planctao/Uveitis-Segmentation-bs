"""Verify and replay the remaining already-owned screening jobs.

This observer never starts or resumes training. ConDSeg and SpectralMamba
have existing observers; this one covers PU, U-KAN, MK and paired controls.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.run_paper_reproduction import atomic_json, process_start, wait_for_external_job
from scripts.verify_paper_experiment import SUMMARY_FIELDS, verify_run


def replay_matches(replay: dict | None, verified: dict) -> bool:
    """Bind a full replay to actual checkpoint, config, history and numbers."""
    if not isinstance(replay, dict):
        return False
    expected = {"run": verified["run"], "fold": verified["fold"], "best_epoch": verified["best_epoch"],
                "checkpoint_sha256": verified["checkpoints"]["best"]["sha256"],
                "resolved_config_sha256": verified["resolved_config_sha256"],
                "metric_history_sha256": verified["metric_history_sha256"], "validation_samples": 444,
                "validation_augmented_excluded": True, "all_required_metrics_match": True, "reported": verified["metrics"]}
    if any(replay.get(key) != value for key, value in expected.items()):
        return False
    try:
        return all(math.isclose(float(replay["measured"][field]), float(verified["metrics"][key]), rel_tol=1e-6, abs_tol=1e-6)
                   for key, field in SUMMARY_FIELDS.items())
    except (KeyError, TypeError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/paper_reproduction_20261011")
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    torch.set_num_threads(2)
    root = Path(args.root).resolve()
    state_path = root / "remaining_screen_verification_state.json"
    with state_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"status": "running", "supervisor_pid": os.getpid(), "supervisor_start": process_start(os.getpid()),
                 "scope": "Observe original secondary-owned jobs; no training launch or restart", "verified": {}}
        atomic_json(state_path, state)
        try:
            for method, candidate in [("puunet", True), ("ukan", True), ("mkunet", True),
                                      ("puunet_control", False), ("spectralmamba_control", False)]:
                name = f"author_{method}_f1_30_256"
                state.update(status="waiting_for_owned_training", pending_job=name)
                atomic_json(state_path, state)
                job = wait_for_external_job(name, root / "secondary_screen_state.json")
                if job.get("return_code") != 0:
                    raise RuntimeError(f"No captured successful training return code: {name}")
                run = Path(job["run"])
                verified = verify_run(run, "f1", 30, require_full_data=True)
                if job["metrics"] != verified["metrics"]:
                    raise RuntimeError(f"Queue metrics differ from actual selected checkpoint: {name}")
                atomic_json(root / "verified_experiments" / f"{name}.json", verified)
                evaluation_path = root / f"{method}_full_checkpoint_evaluation_audit.json"
                # Reuse an existing successful full replay only for this exact
                # checkpoint/history/config, all 444 images and matching values.
                replay = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else None
                valid_replay = replay_matches(replay, verified)
                if not valid_replay:
                    state.update(status="replaying_full_validation", pending_job=name)
                    atomic_json(state_path, state)
                    command = [sys.executable, "scripts/audit_paper_checkpoint_evaluation.py", "--run", str(run),
                               "--fold", "f1", "--output", str(evaluation_path)]
                    with (root / f"{method}_full_checkpoint_evaluation_cli.log").open("a") as stream:
                        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                        state.update(evaluation_pid=child.pid, evaluation_start=process_start(child.pid))
                        atomic_json(state_path, state)
                        code = child.wait()
                    state.pop("evaluation_pid", None)
                    state.pop("evaluation_start", None)
                    if code != 0:
                        raise RuntimeError(f"Full validation replay failed: {name}, code={code}")
                    replay = json.loads(evaluation_path.read_text())
                if not replay_matches(replay, verified):
                    raise RuntimeError(f"Full validation metrics differ: {name}")
                config = json.loads((run / "resolved_config.json").read_text())
                record = {"method": method, "implementation": config["reproduction"]["implementation"],
                          "experiment_class": "FA_30_epoch_screen" if candidate else "FA_matched_skeleton_control",
                          "candidate_result": candidate, "resolution": 256, "fold": "f1", "train_samples": 1965,
                          "clean_validation_samples": replay["validation_samples"], "epochs": 30,
                          "best_epoch": verified["best_epoch"], "encoder_pretrained": config["model"].get("pretrained", False),
                          "model_config": config["model"], "objective_config": config["loss"],
                          "protocol_adaptations": config["reproduction"], "metrics": verified["metrics"],
                          "checkpoint_audit": f"verified_experiments/{name}.json", "checkpoint_sha256": replay["checkpoint_sha256"],
                          "resolved_config_sha256": hashlib.sha256((run / "resolved_config.json").read_bytes()).hexdigest(),
                          "full_validation_replay": evaluation_path.name, "all_replayed_metrics_match": True,
                          "all_replayed_metrics_exact": all(float(replay["measured"][field]) == float(verified["metrics"][key])
                                                            for key, field in SUMMARY_FIELDS.items()),
                          "return_code": 0, "completed_at": job["finished_at"], "audited_at": time.time(),
                          "conclusion": "Verified FA/256/30 result; not original benchmark-score reproduction or matched-768 comparison."}
                record_path = root / (f"completed_{method}_author.json" if candidate else f"completed_{method}.json")
                atomic_json(record_path, record)
                state["verified"][name] = {"record": record_path.name, "checkpoint_sha256": replay["checkpoint_sha256"], "candidate_result": candidate}
                atomic_json(state_path, state)
                print(f"VERIFIED_FULL_SCREEN {name}, shared_macro={verified['metrics']['best_paper_sweep_macro_dice']}", flush=True)
            state.update(status="complete", pending_job=None)
            atomic_json(state_path, state)
        except Exception as error:
            state.update(status="failed", error=str(error))
            atomic_json(state_path, state)
            raise


if __name__ == "__main__":
    main()
