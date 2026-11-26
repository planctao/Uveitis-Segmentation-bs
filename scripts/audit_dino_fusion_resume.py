"""Audit actual DINO+EDG checkpoint resume on a small, explicitly limited FA run.

This checks training lifecycle, weights and optimizer/scheduler consistency.
It is not an accuracy screen and cannot satisfy full-data experiment checks.
"""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.run_paper_reproduction import atomic_json, completed_metrics
from scripts.verify_paper_experiment import verify_run


def compare_nested(a, b, location="state"):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0, msg=lambda message: f"{location}: {message}")
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise AssertionError(f"Keys differ at {location}")
        for key in a:
            compare_nested(a[key], b[key], f"{location}/{key}")
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError(f"Lengths differ at {location}")
        for index, (first, second) in enumerate(zip(a, b)):
            compare_nested(first, second, f"{location}/{index}")
    elif a != b:
        raise AssertionError(f"Value differs at {location}: {a} versus {b}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/edg_validation_20261011/dino_resume_audit_v3")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--interrupt-after-epoch", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.interrupt_after_epoch < args.epochs:
        parser.error("Interruption must occur before the final epoch")
    epochs, interrupt_epoch = args.epochs, args.interrupt_after_epoch
    torch.set_num_threads(2)
    root = (PROJECT_ROOT / args.root).resolve()
    root.relative_to(PROJECT_ROOT / "outputs")
    root.mkdir(parents=True, exist_ok=True)
    paths = ("scripts/train_dino_fusion_paired.py", "scripts/train_dinov3_multilabel.py",
             "scripts/train_unet_multilabel.py", "src/bs/convnext_seg.py", "src/bs/paired_experiments.py",
             "scripts/audit_dino_fusion_resume.py")
    hashes = {path: hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest() for path in paths}
    manifest_path = root / "implementation.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != hashes:
        raise RuntimeError("Audit implementation changed; inspect and use a new audit directory")
    atomic_json(manifest_path, hashes)
    template = PROJECT_ROOT / "outputs/edg_validation_20261011/pending_configs/paired_dino_edg_vsubr_s42_f1_30_768.yaml"
    base = yaml.safe_load(template.read_text())
    run_root = root / "runs"
    configs = {}
    for label in ("uninterrupted", "resumed"):
        config = json.loads(json.dumps(base))
        config["project"]["name"] = f"dino_edg_{label}_audit"
        config["runtime"]["num_workers"] = 0
        config["train"].update(image_size=[64, 64], epochs=epochs, batch_size=2, grad_accum_steps=1,
                               max_train_samples=4, max_val_samples=2, save_interval=epochs,
                               sample_interval=interrupt_epoch)
        config["outputs"]["root"] = str(run_root.relative_to(PROJECT_ROOT))
        config["reproduction"] = {"experiment_class": "lifecycle_audit_only", "input_resolution": 64,
                                  "training_protocol": f"{epochs} epochs / 4 train / 2 validation; actual pretrained DINO+EDG+VS-UBR; interrupt after epoch {interrupt_epoch}",
                                  "deviation": "Explicitly limited data and resolution; never an accuracy result"}
        path = root / f"{label}.yaml"
        content = yaml.safe_dump(config, sort_keys=False)
        if path.exists() and path.read_text() != content:
            raise RuntimeError(f"Existing audit config differs: {path}")
        path.write_text(content)
        configs[label] = path
    environment = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                   "FUSION_AUDIT_INTERRUPT_EPOCH": str(interrupt_epoch)}
    runs = {label: run_root / f"dino_edg_{label}_audit" for label in configs}
    normal = [sys.executable, "scripts/train_dino_fusion_paired.py", "--config", str(configs["uninterrupted"]),
              "--run-name", runs["uninterrupted"].name, "--fold", "f1"]
    if completed_metrics(runs["uninterrupted"], "f1", epochs) is None:
        if runs["uninterrupted"].exists():
            normal += ["--resume", str(runs["uninterrupted"] / "f1/checkpoints/latest.pt")]
        with (root / "uninterrupted.log").open("a") as stream:
            subprocess.run(normal, cwd=PROJECT_ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
    interrupted_code = """
import sys
import os
from scripts import train_dino_fusion_paired as wrapper
wrapper.configure_training()
original_save_samples = wrapper.experiment.save_samples
def stop_after_checkpoint(model, loader, device, fold_dir, epoch, threshold):
    if epoch == int(os.environ['FUSION_AUDIT_INTERRUPT_EPOCH']):
        raise RuntimeError('expected audit interruption after atomic checkpoint')
    return original_save_samples(model, loader, device, fold_dir, epoch, threshold)
wrapper.experiment.save_samples = stop_after_checkpoint
try:
    wrapper.experiment.main()
except RuntimeError as error:
    if str(error) != 'expected audit interruption after atomic checkpoint':
        raise
    print(str(error), flush=True)
else:
    raise RuntimeError('audit did not interrupt')
"""
    interrupted = [sys.executable, "-c", interrupted_code, "--config", str(configs["resumed"]),
                   "--run-name", runs["resumed"].name, "--fold", "f1"]
    if not (runs["resumed"] / "f1/checkpoints/latest.pt").exists():
        with (root / "interrupted.log").open("a") as stream:
            subprocess.run(interrupted, cwd=PROJECT_ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
    interrupted_state_path = root / "interruption_state.json"
    if not interrupted_state_path.exists():
        source_checkpoint = runs["resumed"] / "f1/checkpoints/latest.pt"
        first = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
        if first["epoch"] != interrupt_epoch or first["scheduler"]["last_epoch"] != interrupt_epoch:
            raise RuntimeError("Interruption did not persist completed epoch and advanced scheduler")
        optimizer_state = first["optimizer"]["state"]
        if not optimizer_state:
            raise RuntimeError("Pre-interruption optimizer state is empty; audit must interrupt after actual learning")
        steps = sorted({int(state["step"].item()) for state in optimizer_state.values()})
        if not steps or min(steps) < 1:
            raise RuntimeError("Pre-interruption optimizer counters do not establish actual updates")
        preserved = root / "interruption_checkpoint.pt"
        shutil.copyfile(source_checkpoint, preserved)
        interrupted_state = {"epoch": interrupt_epoch, "optimizer_state_entries": len(optimizer_state),
                             "distinct_parameter_step_counts": steps, "scheduler_last_epoch": first["scheduler"]["last_epoch"],
                             "checkpoint": str(preserved), "checkpoint_sha256": hashlib.sha256(preserved.read_bytes()).hexdigest()}
        atomic_json(interrupted_state_path, interrupted_state)
        del first
    interrupted_state = json.loads(interrupted_state_path.read_text())
    if hashlib.sha256(Path(interrupted_state["checkpoint"]).read_bytes()).hexdigest() != interrupted_state["checkpoint_sha256"]:
        raise RuntimeError("Preserved interruption checkpoint changed")
    if completed_metrics(runs["resumed"], "f1", epochs) is None:
        command = [sys.executable, "scripts/train_dino_fusion_paired.py", "--config", str(configs["resumed"]),
                   "--run-name", runs["resumed"].name, "--fold", "f1", "--resume",
                   str(runs["resumed"] / "f1/checkpoints/latest.pt")]
        with (root / "resumed.log").open("a") as stream:
            subprocess.run(command, cwd=PROJECT_ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
    verified = {label: verify_run(run, "f1", epochs) for label, run in runs.items()}
    a = torch.load(runs["uninterrupted"] / "f1/checkpoints/latest.pt", map_location="cpu", weights_only=False)
    b = torch.load(runs["resumed"] / "f1/checkpoints/latest.pt", map_location="cpu", weights_only=False)
    for field in ("model", "optimizer", "scheduler", "scaler", "epoch", "best_epoch", "best_score", "best_metrics"):
        compare_nested(a[field], b[field], field)
    if not a["optimizer"]["state"]:
        raise RuntimeError("Audit made no optimizer update; cannot establish actual training resume")
    weight_max_delta = max(float((tensor - b["model"][name]).abs().max())
                           for name, tensor in a["model"].items() if tensor.is_floating_point())
    histories = {}
    for label, run in runs.items():
        with (run / "f1/metrics.csv").open() as stream:
            histories[label] = list(csv.DictReader(stream))
    if histories["uninterrupted"] != histories["resumed"]:
        raise RuntimeError("Uninterrupted and resumed metric histories differ")
    executed = [int(float(row["train_executed_optimizer_updates"])) for row in histories["uninterrupted"]]
    planned = [int(float(row["train_planned_optimizer_updates"])) for row in histories["uninterrupted"]]
    if not any(executed[interrupt_epoch:]):
        raise RuntimeError("No actual optimizer update after resume")
    report = {"status": "passed", "epochs": epochs, "train_samples": 4, "validation_samples": 2, "image_size": [64, 64],
              "model_optimizer_scheduler_scaler_match": True, "metric_histories_identical": True,
              "tensor_comparison_tolerances": {"rtol": 0, "atol": 0},
              "interruption_state": interrupted_state, "nonempty_optimizer_restoration_verified": True,
              "maximum_weight_difference": weight_max_delta, "implementation_sha256": hashes,
              "planned_optimizer_updates_per_epoch": planned, "executed_optimizer_updates_per_epoch": executed,
              "checkpoints": {label: verified[label]["checkpoints"] for label in runs},
              "scope": "actual DINO+EDG+VS-UBR lifecycle audit; limited data/resolution; no accuracy claim"}
    atomic_json(root / "resume_audit.json", report)
    print(json.dumps({key: report[key] for key in ("status", "model_optimizer_scheduler_scaler_match", "metric_histories_identical", "maximum_weight_difference")}, indent=2))


if __name__ == "__main__":
    main()
