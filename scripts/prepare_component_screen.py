"""Prepare CAT, MIL and joint lesion2 supervision with the same FA baseline."""

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
    torch.set_num_threads(2)
    root = PROJECT_ROOT / "outputs/overnight_unet_20261012/component_screen"
    root.mkdir(parents=True, exist_ok=True)
    if (root / "queue_state.json").exists():
        raise RuntimeError("Cannot overwrite a launched queue")
    baseline = PROJECT_ROOT / "outputs/edg_validation_20261011/runs/paired_concat_s42_f1_30_256"
    original = json.loads((baseline / "resolved_config.json").read_text())
    reference = json.loads((baseline / "f1/initial_weight_fingerprints.json").read_text())
    jobs, audit = [], {}
    for mode in ["mil", "cat", "joint"]:
        config = copy.deepcopy(original)
        name = f"catmil_lesion2_{mode}_s42_f1_30_256"
        config["project"]["name"] = name
        config["outputs"]["root"] = str(root.relative_to(PROJECT_ROOT) / "runs")
        config["model"]["name"] = "component_supervised_controlled"
        config["model"]["component_supervision"] = {
            "mode": mode, "cat_weight": .1, "mil_weight": .2, "cat_warmup_epochs": 10}
        config["reproduction"] = {
            "experiment_class": "lesion_level_objective_mechanism_screen",
            "paper_url": "https://arxiv.org/abs/2604.08015v3",
            "author_commit": "e81ba4a4759d3804f96e9804d72d5bfa512dd592",
            "author_operator_scope": "binary3D CAT/MIL forward and gradient checked; no whole framework reproduction",
            "deviations": ["2D4-neighbor connected components computed independently per augmented image",
                           "auxiliary applied only to independent lesion2 sigmoid, including palette2 and3 overlap",
                           "original FA segmentation loss retained instead of DiceCE",
                           "CAT warmup10 of30 epochs instead of50 of150, retaining one-third schedule fraction",
                           "MIL max reduction vectorized; ignore pixels excluded"],
            "unchanged": "network, inference, all292 initial tensors, data, optimizer, sampling and original loss/metrics",
            "selection": "original shared-threshold Macro Dice; detection, FP/FN, tiny coverage and negative-image errors are diagnostics",
            "caution": "paper mainly improves lesion recall with FP cost; does not guarantee pooled Dice improvement"}
        for section in ["data", "train", "loss", "metric", "augmentations", "runtime"]:
            assert config[section] == original[section]
        model = initialize_paired_model(build_model, config)
        fingerprints = initial_weight_fingerprints(model)
        assert fingerprints == reference
        audit[name] = {"actual_baseline_initial_tensors_matched": len(reference),
                       "parameters": sum(p.numel() for p in model.parameters()),
                       "all_initial_states_equal_baseline": True}
        path = root / "input_configs" / f"{name}.yaml"
        path.parent.mkdir(exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        jobs.append({"id": name, "config": str(path.relative_to(PROJECT_ROOT)), "fold": "f1",
            "training_script": "scripts/train_component_supervised_multilabel.py",
            "reference_diagnostics": "outputs/edg_validation_20261011/diagnostics/paired_concat_s42_f1_30_256"})
        del model
    atomic_json(root / "plan.json", {"jobs": jobs, "created_epoch": time.time(),
        "advance_rule": "Complete all three30epoch jobs and full replays. Joint must exceed plain and each single term before joint replication. A single term may replicate only if it exceeds plain; retain all failures and inspect negative FP, positive-image Dice and tiny coverage. No strong transfer without repeated positive evidence.",
        "priority": "training-only semantic followup to zero-response lesion diagnostics; admit when512 baseline frees memory and whole-model probe passes",
        "training_started_at_preparation": False})
    sources = ["src/bs/component_supervision.py", "src/bs/component_supervised_unet.py",
               "scripts/train_component_supervised_multilabel.py", "scripts/train_unet_multilabel.py",
               "src/bs/paired_experiments.py", "src/bs/aaag_unet.py", "src/bs/multilabel.py",
               "src/bs/dataset.py", "src/bs/augmentations.py"]
    atomic_json(root / "source_manifest.json", {"created_epoch": time.time(), "source_sha256": {
        name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() for name in sources}})
    atomic_json(root / "preparation_audit.json", {"status": "prepared_not_launched", "jobs": audit,
        "unit_tests": "7 passed; per-image labeling, inverse-size formula, MIL bag gradients, ignore/empty/clipping, exact state/inference, warmup and overlap gradients",
        "operator_audit": "outputs/overnight_unet_20261012/component_semantic_preparation/operator_audit.json"})
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
