"""Prepare new seeds only after the complete CATMIL screen passes its gate."""

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
from scripts.train_component_supervised_multilabel import build_model
from scripts.run_paper_reproduction import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=["cat", "mil", "joint"], required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    overnight = PROJECT_ROOT / "outputs/overnight_unet_20261012"
    screen = overnight / "component_screen"
    queue = json.loads((screen / "queue_state.json").read_text())
    diagnostics = json.loads((screen / "diagnostic_state.json").read_text())
    if queue["status"] != "complete" or diagnostics["status"] != "complete":
        raise RuntimeError("All initial screen arms and full diagnostics must finish first")
    scores = {}
    recipes = {}
    for mode in ["mil", "cat", "joint"]:
        name = f"catmil_lesion2_{mode}_s42_f1_30_256"
        job = queue["jobs"][name]
        if job["status"] != "complete" or int(job["epochs"]) != 30:
            raise RuntimeError("Require full30epoch initial screen")
        replay = json.loads((screen / "verified_experiments" / f"{name}_replay.json").read_text())
        if replay["validation_samples"] != 444 or not replay["all_required_metrics_match"]:
            raise RuntimeError("Initial screen lacks complete clean-fold replay")
        scores[mode] = job["metrics"]["best_paper_sweep_macro_dice"]
        recipes[mode] = json.loads((Path(job["run"]) / "resolved_config.json").read_text())
    reference_root = PROJECT_ROOT / "outputs/edg_validation_20261011"
    scores["plain"] = json.loads((reference_root / "diagnostics/paired_concat_s42_f1_30_256/summary.json").read_text())["reported_macro"]["shared"]
    controls = ["plain", "cat", "mil"] if args.candidate == "joint" else ["plain"]
    if not all(scores[args.candidate] > scores[control] for control in controls):
        raise RuntimeError(f"Original complete-screen gate not met: {scores}")
    root = overnight / "component_replication"
    root.mkdir(exist_ok=True)
    if (root / "plan.json").exists() or (root / "queue_state.json").exists():
        raise RuntimeError("Cannot overwrite an existing replication plan")
    modes = ["joint", "cat", "mil"] if args.candidate == "joint" else [args.candidate]
    jobs, audits = [], {}
    for seed in [2026, 3407]:
        baseline = reference_root / "runs" / f"paired_concat_s{seed}_f1_30_256"
        original = json.loads((baseline / "resolved_config.json").read_text())
        reference = json.loads((baseline / "f1/initial_weight_fingerprints.json").read_text())
        for mode in modes:
            config = copy.deepcopy(original)
            name = f"catmil_lesion2_{mode}_s{seed}_f1_30_256"
            config["project"]["name"] = name
            config["outputs"]["root"] = str(root.relative_to(PROJECT_ROOT) / "runs")
            config["model"]["name"] = "component_supervised_controlled"
            config["model"]["component_supervision"] = copy.deepcopy(recipes[mode]["model"]["component_supervision"])
            config["reproduction"] = copy.deepcopy(recipes[mode]["reproduction"])
            config["reproduction"].update(experiment_class="additional_seed_paired_component_replication",
                                           seed42_completed_gate_scores=scores,
                                           candidate=args.candidate,
                                           seed=seed)
            for section in ["data", "train", "loss", "metric", "augmentations", "runtime"]:
                if config[section] != original[section]:
                    raise RuntimeError(f"Recipe changed: {section}")
            model = initialize_paired_model(build_model, config)
            actual = initial_weight_fingerprints(model)
            if actual != reference:
                raise RuntimeError("Original same-seed initial tensors differ")
            audits[name] = {"original_initial_tensors_matched": len(reference),
                           "parameter_count": sum(p.numel() for p in model.parameters()),
                           "same_seed_original_baseline": str(baseline),
                           "auxiliary_settings_unchanged_from_seed42": True}
            path = root / "input_configs" / f"{name}.yaml"
            path.parent.mkdir(exist_ok=True)
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            jobs.append({"id": name, "config": str(path.relative_to(PROJECT_ROOT)), "fold": "f1",
                         "training_script": "scripts/train_component_supervised_multilabel.py",
                         "comparison_group": str(seed),
                         "reference_diagnostics": str((reference_root / "diagnostics" / f"paired_concat_s{seed}_f1_30_256").relative_to(PROJECT_ROOT))})
            del model
    atomic_json(root / "plan.json", {"jobs": jobs, "created_epoch": time.time(),
        "candidate": args.candidate, "complete_seed42_scores": scores,
        "advance_rule": "Candidate must exceed same-seed plain for all3 seeds after complete30epoch runs/replays; if joint, also exceed same-seed CAT and MIL in all3 seeds. Keep failures, inspect pixel/instance/rare-lesion diagnostics; no formal strong transfer without this repeated evidence and enough full paired-training window.",
        "expected_initial_difference_suffixes": [],
        "training_started_at_preparation": False})
    screen_manifest = json.loads((screen / "source_manifest.json").read_text())
    sources = list(screen_manifest["source_sha256"])
    for name, expected in screen_manifest["source_sha256"].items():
        if hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Screen training source changed: {name}")
    sources += [job["config"] for job in jobs] + [str((root / "plan.json").relative_to(PROJECT_ROOT))]
    atomic_json(root / "source_manifest.json", {"created_epoch": time.time(), "source_sha256": {
        name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() for name in sources}})
    atomic_json(root / "preparation_audit.json", {"status": "prepared_not_launched", "jobs": audits,
        "candidate": args.candidate, "complete_seed42_scores": scores,
        "scope": "Same-seed original initialization/recipe and seed42 auxiliary settings; no training or FA accuracy result from preparation"})
    print(json.dumps({"candidate": args.candidate, "jobs": audits, "complete_seed42_scores": scores}, indent=2))


if __name__ == "__main__":
    main()
