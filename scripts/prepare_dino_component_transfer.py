"""Prepare a gated, full-budget DINO512 component-supervision candidate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from bs.paired_experiments import initialize_paired_model, initial_weight_fingerprints
from scripts.run_paper_reproduction import atomic_json
from scripts.train_dino_component_supervised import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["cat", "mil", "joint"], default="cat")
    args = parser.parse_args()
    torch.set_num_threads(2)
    overnight = PROJECT_ROOT / "outputs/overnight_unet_20261012"
    baseline_root = overnight / "strong_baseline_512"
    baseline = baseline_root / "runs/dino_baseline_vsubr_s42_f1_30_512"
    queue = json.loads((baseline_root / "queue_state.json").read_text())
    replay = json.loads((baseline_root / "verified_experiments/dino_baseline_vsubr_s42_f1_30_512_replay.json").read_text())
    if queue["status"] != "complete" or replay["validation_samples"] != 444 or not replay["all_required_metrics_match"]:
        raise RuntimeError("Require the completed, replayed DINO512 control")
    original = json.loads((baseline / "resolved_config.json").read_text())
    root = overnight / "dino_component_transfer_512"
    root.mkdir(exist_ok=True)
    if (root / "plan.json").exists() or (root / "queue_state.json").exists():
        raise RuntimeError("Do not overwrite a prepared or running candidate")
    name = f"dino_catmil_{args.mode}_vsubr_s42_f1_30_512"
    config = copy.deepcopy(original)
    config["project"]["name"] = name
    config["outputs"]["root"] = str(root.relative_to(PROJECT_ROOT) / "runs")
    screen_root = overnight / "component_screen"
    screen_recipe = json.loads((screen_root / f"runs/catmil_lesion2_{args.mode}_s42_f1_30_256/resolved_config.json").read_text())
    config["model"]["training_component_supervision"] = copy.deepcopy(screen_recipe["model"]["component_supervision"])
    config["reproduction"].update(experiment_class="conditional_component_objective_strong_transfer",
        paper_url=screen_recipe["reproduction"]["paper_url"],
        author_commit=screen_recipe["reproduction"]["author_commit"],
        mode=args.mode, original_strong_control=str(baseline),
        deviations=screen_recipe["reproduction"]["deviations"],
        transfer_scope="Training objective only; original pretrained backbone/random decoder initialization, image-aware VS-UBR loss, single-output model and full30epoch512 recipe",
        admission_rule="Complete three-seed positive mechanism evidence and full replay, real CUDA batch8 optimizer-memory probe, and enough measured time for complete30epoch512 training. If joint, matched single-term strong controls are additionally required.")
    for section in ["data", "train", "loss", "metric", "augmentations", "runtime"]:
        if config[section] != original[section]:
            raise RuntimeError(f"Strong recipe changed: {section}")
    original_model = copy.deepcopy(config["model"])
    original_model.pop("training_component_supervision")
    if original_model != original["model"]:
        raise RuntimeError("Baseline architecture changed")
    model = initialize_paired_model(build_model, config)
    actual = initial_weight_fingerprints(model)
    reference = json.loads((baseline / "f1/initial_weight_fingerprints.json").read_text())
    if actual != reference:
        raise RuntimeError("Original DINO512 initialization differs")
    parameters = sum(p.numel() for p in model.parameters())
    path = root / "input_configs" / f"{name}.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    atomic_json(root / "plan.json", {"jobs": [{"id": name, "config": str(path.relative_to(PROJECT_ROOT)),
        "fold": "f1", "training_script": "scripts/train_dino_component_supervised.py",
        "reference_diagnostics": str((baseline_root / "diagnostics/dino_baseline_vsubr_s42_f1_30_512").relative_to(PROJECT_ROOT))}],
        "created_epoch": time.time(), "mode": args.mode, "admission_rule": config["reproduction"]["admission_rule"],
        "expected_initial_difference_suffixes": [], "training_started_at_preparation": False})
    manifest = json.loads((baseline_root / "source_manifest.json").read_text())
    for name_, expected in manifest["source_sha256"].items():
        if hashlib.sha256((PROJECT_ROOT / name_).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Original source changed: {name_}")
    sources = list(manifest["source_sha256"]) + ["src/bs/component_supervision.py", "src/bs/component_objective.py",
        "src/bs/paired_experiments.py", "scripts/train_dino_component_supervised.py",
        str(path.relative_to(PROJECT_ROOT)), str((root / "plan.json").relative_to(PROJECT_ROOT))]
    atomic_json(root / "source_manifest.json", {"created_epoch": time.time(), "source_sha256": {
        name_: hashlib.sha256((PROJECT_ROOT / name_).read_bytes()).hexdigest() for name_ in sources}})
    report = {"status": "prepared_not_admitted_or_launched", "mode": args.mode,
        "all_original_initial_tensors_matched": len(reference), "parameter_count": parameters,
        "same_architecture_and_full_training_recipe": True,
        "same_original_component_weights_and_warmup": True,
        "cuda_probe_passed": False, "three_seed_gate_passed": False,
        "scope": "Preparation and actual initialization audit only; no accuracy training or score"}
    atomic_json(root / "preparation_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
