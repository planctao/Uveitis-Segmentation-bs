"""Single-mechanism Stair Pooling screen after the paired EDG training queue."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.run_paper_reproduction import atomic_json, process_alive, process_start, run_plan
from scripts.compare_fusion_diagnostics import paired_comparison
from scripts.verify_paper_experiment import verify_run


def prepare(root: Path, dependency_root: Path) -> Path:
    source = dependency_root / "input_configs/paired_concat_s42_f1_30_256.yaml"
    base = yaml.safe_load(source.read_text())
    config_dir = root / "input_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for method in ("maxpool", "capacity", "stair"):
        name = f"pool_{method}_s42_f1_30_256"
        config = json.loads(json.dumps(base))
        config["project"]["name"] = name
        config["model"].update(name="stair_pooling_controlled", downsampling=method, pooling_stages=[1, 2])
        config["outputs"]["root"] = str(root.relative_to(PROJECT_ROOT) / "runs")
        config["reproduction"] = {
            "experiment_class": "single_mechanism_screen", "paper_method": "stair_pooling",
            "implementation": "Eq2-4_two_directional_paths_on_first_two_nnunet_reductions",
            "input_resolution": 256,
            "deviation": "paper describes max pooling; nnU-Net strided convolutions are changed to stride1 after pooling; 3x3 branch convolutions and 1x1 fusion explicitly chosen; no transfer entropy selection",
            "controls": "matched ordinary maxpool and same branch parameter count after one 2x2 maxpool; paired strided nnU-Net from EDG screen",
        }
        path = config_dir / f"{name}.yaml"
        text = yaml.safe_dump(config, sort_keys=False)
        if path.exists() and path.read_text() != text:
            raise RuntimeError(f"Experiment config changed: {path}")
        if not path.exists():
            path.write_text(text)
        jobs.append({"id": name, "config": str(path.relative_to(PROJECT_ROOT)), "fold": "f1",
                     "training_script": "scripts/train_stair_pooling_multilabel.py"})
    path = root / "plan.json"
    plan = {"seed": 42, "jobs": jobs, "dependency_root": str(dependency_root),
            "strided_control": "paired_concat_s42_f1_30_256"}
    if path.exists() and json.loads(path.read_text()) != plan:
        raise RuntimeError("Stair experiment plan changed")
    if not path.exists():
        atomic_json(path, plan)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/stair_validation_20261011")
    parser.add_argument("--dependency-root", default="outputs/edg_validation_20261011")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root, dependency = (PROJECT_ROOT / args.root).resolve(), (PROJECT_ROOT / args.dependency_root).resolve()
    root.relative_to(PROJECT_ROOT / "outputs")
    root.mkdir(parents=True, exist_ok=True)
    with (root / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare(root, dependency)
        if args.prepare_only:
            print(plan)
            return
        status_path = root / "status.json"
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        if previous.get("status") == "complete":
            print("Stair screen already complete")
            return
        paths = ("src/bs/stair_pooling.py", "scripts/train_stair_pooling_multilabel.py", "scripts/train_unet_multilabel.py", "src/bs/paired_experiments.py")
        hashes = {name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() for name in paths}
        if previous.get("source_sha256", hashes) != hashes:
            raise RuntimeError("Stair training implementation changed since scheduling")
        status = {"status": "waiting_for_paired_training", "supervisor_pid": os.getpid(),
                  "supervisor_start": process_start(os.getpid()), "source_sha256": hashes, "plan": str(plan)}
        atomic_json(status_path, status)
        try:
            while True:
                training = json.loads((dependency / "paired_queue_state.json").read_text())
                if training["status"] == "complete":
                    break
                if training["status"] == "failed":
                    raise RuntimeError("Paired EDG training queue failed; diagnose before starting extra experiments")
                if not process_alive(training["supervisor_pid"], training.get("supervisor_start")):
                    raise RuntimeError("Paired training supervisor missing; recover it first")
                time.sleep(30)
            status["status"] = "training"
            atomic_json(status_path, status)
            state_path = root / "queue_state.json"
            run_plan(plan, state_path, resume=True)
            state = json.loads(state_path.read_text())
            status["status"] = "auditing"
            atomic_json(status_path, status)
            fingerprints = {}
            summaries = {}
            for method in ("maxpool", "capacity", "stair"):
                name = f"pool_{method}_s42_f1_30_256"
                run = Path(state["jobs"][name]["run"])
                verified = verify_run(run, "f1", 30, require_full_data=True)
                atomic_json(root / "verified_experiments" / f"{name}.json", verified)
                fingerprints[method] = json.loads((run / "f1/initial_weight_fingerprints.json").read_text())
                output = root / "diagnostics" / method
                subprocess.run([sys.executable, "scripts/analyze_fusion_checkpoint.py", "--run", str(run),
                                "--output", str(output), "--save-probabilities"], cwd=PROJECT_ROOT, check=True)
                summaries[method] = json.loads((output / "summary.json").read_text())
                subprocess.run([sys.executable, "scripts/audit_paper_checkpoint_evaluation.py", "--run", str(run),
                                "--output", str(root / "verified_experiments" / f"{name}_replay.json")], cwd=PROJECT_ROOT, check=True)
            if fingerprints["stair"] != fingerprints["capacity"]:
                raise RuntimeError("Stair and capacity-control initial weights differ")
            baseline_run = Path(training["jobs"]["paired_concat_s42_f1_30_256"]["run"])
            baseline_fingerprints = json.loads((baseline_run / "f1/initial_weight_fingerprints.json").read_text())
            for method, weights in fingerprints.items():
                if any(weights.get(name) != value for name, value in baseline_fingerprints.items()):
                    raise RuntimeError(f"Shared baseline weights differ for {method}")
            baseline_diagnostics = dependency / "diagnostics/paired_concat_s42_f1_30_256"
            if not (baseline_diagnostics / "summary.json").exists():
                raise RuntimeError("Paired baseline diagnostics pending; recover the diagnostic observer")
            comparisons = {method: paired_comparison(root / "diagnostics/stair", root / "diagnostics" / method)
                           for method in ("maxpool", "capacity")}
            comparisons["strided"] = paired_comparison(root / "diagnostics/stair", baseline_diagnostics)
            advance = all(value["comparisons"]["shared"]["macro_delta"] > 0 for value in comparisons.values())
            result = {"comparisons": comparisons, "initialization_verified": True,
                      "advance_to_more_seeds": advance,
                      "scope": "seed42/f1/256 single-mechanism exploration; no EDG combination trained and no independent-test gain claimed"}
            atomic_json(root / "results.json", result)
            status.update(status="complete", result=str(root / "results.json"), advance_to_more_seeds=advance)
            atomic_json(status_path, status)
        except Exception as error:
            status.update(status="failed", error=str(error))
            atomic_json(status_path, status)
            raise


if __name__ == "__main__":
    main()
