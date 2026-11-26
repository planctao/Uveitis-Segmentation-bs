"""Durable sequential runner for full paper architectures on FA data.

Uses a lock, PID start times, subprocess return codes, atomically saved state,
terminal epoch checks, and finite metric validation. Observation timeouts never
restart a live job. A failed job stops the queue for diagnosis; --resume resumes
its last atomic checkpoint and never overwrites a completed run.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def process_start(pid: int) -> str | None:
    try:
        # The command in parentheses may contain spaces: split after its end.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def process_alive(pid: int, start: str | None) -> bool:
    return start is not None and process_start(pid) == start


def completed_metrics(run: Path, fold: str, epochs: int) -> dict[str, Any] | None:
    summary_path, metrics_path = run / "summary.json", run / fold / "metrics.csv"
    if not summary_path.exists() or not metrics_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text())
        selected = next(row for row in summary["folds"] if row["fold"] == fold)
        with metrics_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if [int(row["epoch"]) for row in rows] != list(range(1, epochs + 1)):
            return None
        for row in rows:
            if any(not math.isfinite(float(value)) for key, value in row.items() if key != "epoch"):
                return None
        if not all(math.isfinite(float(v)) for k, v in selected.items() if "dice" in k or "loss" in k):
            return None
        if not (run / fold / "checkpoints" / "latest.pt").exists():
            return None
        return selected
    except (KeyError, StopIteration, ValueError, TypeError, json.JSONDecodeError):
        return None


def prepare_resume(run: Path, fold: str) -> Path:
    import torch
    latest = run / fold / "checkpoints" / "latest.pt"
    if not latest.exists():
        raise RuntimeError(f"Incomplete run has no atomic latest checkpoint: {run}; inspect its log")
    checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
    epoch = int(checkpoint["epoch"])
    path = run / fold / "metrics.csv"
    if path.exists():
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            rows = [row for row in reader if int(row["epoch"]) <= epoch]
        if [int(row["epoch"]) for row in rows] != list(range(1, epoch + 1)):
            raise RuntimeError(f"Checkpoint and metric history disagree: {run}")
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    return latest


def wait_for_external_job(name: str, state_path: Path) -> dict[str, Any]:
    """Observe the other owned worker, including jobs not started there yet.

    The external plan owns these jobs exclusively. A polling interval is
    never interpreted as process termination or permission to launch a copy.
    """
    while True:
        if not state_path.exists():
            raise RuntimeError(f"External worker state does not exist: {state_path}")
        external = json.loads(state_path.read_text())
        job = external["jobs"].get(name)
        if job is not None and job["status"] == "complete":
            return job
        if external["status"] == "failed" or (job and job["status"] == "failed"):
            raise RuntimeError(f"External worker failed while executing {name}; inspect {state_path}")
        live_supervisor = process_alive(external["supervisor_pid"], external.get("supervisor_start"))
        live_child = bool(job and job.get("pid") and process_alive(job["pid"], job.get("process_start")))
        if not live_supervisor and not live_child:
            raise RuntimeError(f"External worker and requested child are missing: {name}; recover that worker")
        print(f"waiting for externally owned job {name}", flush=True)
        time.sleep(20)


def run_plan(plan_path: Path, state_path: Path, resume: bool = False,
             external_plan: Path | None = None, external_state: Path | None = None) -> None:
    os.chdir(PROJECT_ROOT)
    plan = json.loads(plan_path.read_text())
    state = json.loads(state_path.read_text()) if state_path.exists() else {"jobs": {}}
    plan_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    if state.get("plan_sha256", plan_hash) != plan_hash:
        raise RuntimeError("Plan changed after queue start; use a separate state file for another plan")
    state.update(supervisor_pid=os.getpid(), supervisor_start=process_start(os.getpid()),
                 plan=str(plan_path), plan_sha256=plan_hash, status="running")
    atomic_json(state_path, state)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       TORCH_HOME=str(PROJECT_ROOT / "outputs/paper_reproduction_20261011/weights/torch"),
                       HF_HOME=str(PROJECT_ROOT / "outputs/paper_reproduction_20261011/weights/hf"))
    generated = state_path.parent / "queue_configs"
    generated.mkdir(exist_ok=True)
    external_jobs = {j["id"]: j for j in json.loads(external_plan.read_text())["jobs"]} if external_plan else {}
    if external_jobs and external_state is None:
        raise ValueError("An external plan requires its state path")

    for job in plan["jobs"]:
        name, fold = job["id"], job.get("fold", "f1")
        config = yaml.safe_load((PROJECT_ROOT / job["config"]).read_text())
        config["train"]["folds_to_run"] = [fold]
        for section, values in job.get("overrides", {}).items():
            config.setdefault(section, {}).update(values)
        epochs = int(config["train"]["epochs"])
        run = PROJECT_ROOT / config["outputs"]["root"] / name
        previous = state["jobs"].get(name, {})
        if name in external_jobs:
            if external_jobs[name] != job:
                raise RuntimeError(f"External job specification differs from primary plan: {name}")
            external_job = wait_for_external_job(name, external_state)
            result = completed_metrics(run, fold, epochs)
            if result is None or external_job.get("return_code") != 0:
                raise RuntimeError(f"External training result is incomplete or invalid: {name}")
            state["jobs"][name] = {**external_job, "metrics": result, "external_state": str(external_state)}
            atomic_json(state_path, state)
            print(f"verified external complete {name}", flush=True)
            continue
        # A recovered supervisor observes an existing child until it exits.
        # It never assumes a 20-second observation timeout means job failure.
        if previous.get("status") == "running":
            while process_alive(previous["pid"], previous.get("process_start")):
                print(f"observing existing child {name}, pid={previous['pid']}", flush=True)
                time.sleep(20)
        result = completed_metrics(run, fold, epochs)
        if result is not None:
            state["jobs"][name] = {**previous, "status": "complete", "metrics": result, "run": str(run)}
            atomic_json(state_path, state)
            print(f"verified complete {name}", flush=True)
            continue
        if job.get("stage1_job"):
            dependency = state["jobs"].get(job["stage1_job"], {})
            if dependency.get("status") != "complete":
                raise RuntimeError(f"Stage 1 dependency is incomplete: {job['stage1_job']}")
            config["model"]["stage1_checkpoint"] = str(Path(dependency["run"]) / fold / "checkpoints" / "best.pt")
        config_path = generated / f"{name}.yaml"
        if run.exists() and (run / "resolved_config.json").exists():
            previous_config = json.loads((run / "resolved_config.json").read_text())
            for section in ("model", "loss", "data", "train"):
                if previous_config[section] != config[section]:
                    raise RuntimeError(f"Refusing to resume with a changed {section}: {name}")
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        training_script = job.get("training_script", "scripts/train_unet_multilabel.py")
        command = [sys.executable, training_script, "--config", str(config_path),
                   "--run-name", name, "--fold", fold]
        if run.exists():
            if not resume:
                raise RuntimeError(f"Incomplete run exists: {name}; inspect and rerun supervisor with --resume")
            command += ["--resume", str(prepare_resume(run, fold))]
        log_path = state_path.parent / f"queue_{name}.log"
        with log_path.open("a") as stream:
            child = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            state["jobs"][name] = {"status": "running", "pid": child.pid, "process_start": process_start(child.pid),
                                    "run": str(run), "log": str(log_path), "command": command,
                                    "started_at": time.time(), "epochs": epochs, "fold": fold}
            atomic_json(state_path, state)
            print(f"started {name}, pid={child.pid}", flush=True)
            code = child.wait()
        result = completed_metrics(run, fold, epochs)
        status = "complete" if code == 0 and result is not None else "failed"
        state["jobs"][name].update(status=status, return_code=code, finished_at=time.time(), metrics=result)
        atomic_json(state_path, state)
        print(f"{status} {name}, return_code={code}, metrics={result}", flush=True)
        if status != "complete":
            state["status"] = "failed"
            atomic_json(state_path, state)
            raise RuntimeError(f"Failed or invalid training run {name}: {log_path}")
    state["status"] = "complete"
    atomic_json(state_path, state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--external-plan", default=None)
    parser.add_argument("--external-state", default=None)
    args = parser.parse_args()
    path = Path(args.state).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_plan(Path(args.plan).resolve(), path, args.resume,
                 Path(args.external_plan).resolve() if args.external_plan else None,
                 Path(args.external_state).resolve() if args.external_state else None)


if __name__ == "__main__":
    main()
