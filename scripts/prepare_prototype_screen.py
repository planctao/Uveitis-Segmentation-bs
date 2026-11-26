"""Prepare teacher-free SWARD PDR arms without starting accuracy training."""

from __future__ import annotations

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
from scripts.train_prototype_regularized_multilabel import build_model
from scripts.run_paper_reproduction import atomic_json


def main():
    torch.set_num_threads(2)
    root = PROJECT_ROOT / "outputs/overnight_unet_20261012/prototype_screen"
    root.mkdir(parents=True, exist_ok=True)
    if (root / "queue_state.json").exists():
        raise RuntimeError("Preparation cannot overwrite a launched queue")
    reference = PROJECT_ROOT / "outputs/edg_validation_20261011/runs/paired_concat_s42_f1_30_256"
    original = json.loads((reference / "resolved_config.json").read_text())
    fingerprints = json.loads((reference / "f1/initial_weight_fingerprints.json").read_text())
    jobs, audits, states = [], {}, []
    for mode in ["joint", "separation", "compactness"]:
        name = f"sward_pdr_{mode}_unit_s42_f1_30_256"
        config = copy.deepcopy(original)
        config["project"]["name"] = name
        config["outputs"]["root"] = str(root.relative_to(PROJECT_ROOT) / "runs")
        config["model"]["name"] = "prototype_regularized_controlled"
        config["model"]["prototype_regularization"] = {
            "mode": mode, "weight": .05, "margin": 1.3, "normalize_features": True}
        config["reproduction"] = {
            "experiment_class": "teacher_free_equation_based_mechanism_screen",
            "paper_url": "https://arxiv.org/abs/2606.00999v2",
            "paper_method": "SWARD printed PDR equations10-13; no MWAD or teacher",
            "deviations": ["four palette-state prototypes including overlap",
                           "full-resolution final decoder prelogit features instead of encoder features",
                           "per-pixel unit feature normalization absent from printed equations",
                           "fixed auxiliary weight0.05 rather than paper gamma1, chosen from training-only gradient scale"],
            "controls": "original paired nnU-Net, separation-only and compactness-only; unchanged inference",
            "selection": "original shared-threshold macro; all444 clean validation images retained",
            "scope": "isolates a prospective FA adaptation; no PDR-only author ablation or author-code parity"}
        for section in ["data", "train", "loss", "metric", "augmentations", "runtime"]:
            if config[section] != original[section]:
                raise RuntimeError(f"Unexpected recipe change: {section}")
        model = initialize_paired_model(build_model, config)
        actual = initial_weight_fingerprints(model)
        assert actual == fingerprints, "All original tensors must match actual saved initial state"
        states.append(actual)
        audits[name] = {"original_tensors_matched": len(actual),
                       "parameters": sum(p.numel() for p in model.parameters()),
                       "baseline_state_keys_unchanged": True,
                       "fixed_auxiliary_weight": .05,
                       "recipe_sections_unchanged": ["data", "train", "loss", "metric", "augmentations", "runtime"]}
        config_path = root / "input_configs" / f"{name}.yaml"
        config_path.parent.mkdir(exist_ok=True)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        jobs.append({"id": name, "config": str(config_path.relative_to(PROJECT_ROOT)), "fold": "f1",
                     "training_script": "scripts/train_prototype_regularized_multilabel.py",
                     "reference_diagnostics": "outputs/edg_validation_20261011/diagnostics/paired_concat_s42_f1_30_256"})
        del model
    assert all(s == states[0] for s in states)
    atomic_json(root / "plan.json", {"jobs": jobs, "created_epoch": time.time(),
        "advance_rule": "Joint shared Macro must exceed original baseline and both single-term controls after full30epoch training/replays; inspect FP/FN, tiny coverage, class eligibility and auxiliary traces before replication. No strong transfer without repeated positive evidence.",
        "no_accuracy_training_started_at_preparation": True,
        "priority": "fallback after current supported-mechanism experiments; do not crowd out matched strong transfer"})
    sources = ["src/bs/prototype_regularization.py", "src/bs/prototype_regularized_unet.py",
               "scripts/train_prototype_regularized_multilabel.py", "scripts/train_unet_multilabel.py",
               "src/bs/paired_experiments.py", "src/bs/aaag_unet.py", "src/bs/multilabel.py",
               "src/bs/dataset.py", "src/bs/augmentations.py"]
    atomic_json(root / "source_manifest.json", {"created_epoch": time.time(), "source_sha256": {
        name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() for name in sources}})
    atomic_json(root / "preparation_audit.json", {"status": "prepared_not_launched", "jobs": audits,
        "all_arms_actual_initial_states_identical": True,
        "baseline": str(reference), "trained_weight_initialization": False,
        "unit_tests": "8 passed; exact model state and inference equivalence included",
        "numerical_probe": "outputs/overnight_unet_20261012/prototype_semantic_preparation/pdr_training_image_numerical_probe.json"})
    atomic_json(root / "weight_decision.json", {"decided_epoch": time.time(), "accuracy_training_started": False,
        "weight": .05, "same_weight_all_arms": True, "paper_gamma": 1.,
        "basis": "training-only initial/epoch29 feature and representative parameter gradient probes; no validation scores used",
        "initial_unweighted_parameter_gradient_ratios": {"encoder_stem": 18.4235363,
            "encoder_deepest": 41.7157936, "decoder_final_feature_conv": 10.4265089},
        "initial_weighted_parameter_gradient_ratios": {"encoder_stem": .921176815,
            "encoder_deepest": 2.08578968, "decoder_final_feature_conv": .521325445},
        "not_a_weight_optimum_or_full_parameter_gradient_balance_claim": True,
        "evidence": "outputs/overnight_unet_20261012/prototype_semantic_preparation/parameter_gradient_probe.json"})
    print(json.dumps(audits, indent=2))


if __name__ == "__main__":
    main()
