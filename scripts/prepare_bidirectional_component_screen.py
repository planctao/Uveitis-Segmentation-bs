"""Prepare an independent BiCC3-arm screen without altering existing queues."""

import copy
import argparse
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
from scripts.train_bidirectional_component_multilabel import build_model
from scripts.run_paper_reproduction import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", type=float, default=1.)
    args = parser.parse_args()
    if args.weight < 0:
        parser.error("weight must be nonnegative")
    torch.set_num_threads(2)
    root = PROJECT_ROOT / "outputs/overnight_unet_20261012/bicc_screen"
    root.mkdir(parents=True, exist_ok=True)
    if (root / "queue_state.json").exists():
        raise RuntimeError("Cannot overwrite a launched queue")
    baseline = PROJECT_ROOT / "outputs/edg_validation_20261011/runs/paired_concat_s42_f1_30_256"
    original = json.loads((baseline / "resolved_config.json").read_text())
    reference = json.loads((baseline / "f1/initial_weight_fingerprints.json").read_text())
    jobs, audit = [], {}
    for mode in ["bidirectional", "gt_only", "smoothed_pred"]:
        config = copy.deepcopy(original)
        name = f"bicc_lesion2_{mode}_s42_f1_30_256"
        config["project"]["name"] = name
        config["outputs"]["root"] = str(root.relative_to(PROJECT_ROOT) / "runs")
        config["model"]["name"] = "bidirectional_component_controlled"
        config["model"]["bidirectional_component_supervision"] = {"mode": mode, "weight": args.weight}
        config["reproduction"] = {
            "experiment_class": "bidirectional_component_objective_mechanism_screen",
            "paper_url": "https://arxiv.org/abs/2609.30223v1",
            "author_commit": "3cf57118746715119366587c720b10ffcc3d975d",
            "paper_default_alpha": .5, "paper_auxiliary_weight": 1., "FA_auxiliary_weight": args.weight,
            "operator_evidence": "outputs/overnight_unet_20261012/bicc_semantic_preparation/operator_audit.json",
            "deviations": ["auxiliary only on independent lesion2 sigmoid, preserving palette2/3 overlap",
                "2D8-neighbor Voronoi cells with isotropic augmented pixel coordinates instead of physical3D spacing",
                "original FA global segmentation objective retained instead of global DiceCE",
                "ignored pixels excluded from component anchors, counts, regional losses and fallback",
                "SciPy CPU labeling/EDT with CUDA differentiable reductions instead of author CuPy backend",
                "no deep supervision;30 epochs instead of500"],
            "control": "gt_only alpha0; smoothed_pred alpha0.5 with pred Dice epsilon1e-5 replacing detached FP score",
            "unchanged": "network, all292 initial tensors, inference, data, sampler, augmentations, optimizer and primary metrics",
            "motivation": "MIL complete30epoch diagnostics increased predicted fragments and worsened one-to-one F1; independent BiCC not combined with MIL",
            "caution": "paper Dice below DiceCE on4of5 datasets; FP/F1 gains do not guarantee primary Dice gain"}
        for section in ["data", "train", "loss", "metric", "augmentations", "runtime"]:
            assert config[section] == original[section]
        model = initialize_paired_model(build_model, config)
        fingerprints = initial_weight_fingerprints(model)
        assert fingerprints == reference
        audit[name] = {"actual_original_initial_tensors_matched": len(reference),
                       "parameters": sum(p.numel() for p in model.parameters()), "all_initial_states_equal_plain": True}
        path = root / "input_configs" / f"{name}.yaml"
        path.parent.mkdir(exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        jobs.append({"id": name, "config": str(path.relative_to(PROJECT_ROOT)), "fold": "f1",
            "training_script": "scripts/train_bidirectional_component_multilabel.py",
            "reference_diagnostics": "outputs/edg_validation_20261011/diagnostics/paired_concat_s42_f1_30_256"})
        del model
    atomic_json(root / "plan.json", {"jobs": jobs, "created_epoch": time.time(),
        "advance_rule": "Complete all three30epoch jobs and full replays. Bidirectional shared Macro must exceed plain, GT-only and smoothed-pred controls before replication; inspect component FP/F1, positive-image Dice and tiny coverage. No formal strong transfer without additional positive seed controls; retain failures unchanged.",
        "priority": "independent FP-supervision followup to completed MIL diagnostics; admit after512 baseline frees memory and complete real-batch CUDA probe passes",
        "training_started_at_preparation": False})
    sources = ["src/bs/bidirectional_component_supervision.py", "src/bs/bidirectional_component_unet.py",
               "scripts/train_bidirectional_component_multilabel.py", "scripts/train_unet_multilabel.py",
               "src/bs/paired_experiments.py", "src/bs/aaag_unet.py", "src/bs/multilabel.py",
               "src/bs/dataset.py", "src/bs/augmentations.py"]
    atomic_json(root / "source_manifest.json", {"created_epoch": time.time(), "source_sha256": {
        name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() for name in sources}})
    atomic_json(root / "preparation_audit.json", {"status": "prepared_not_launched", "jobs": audit,
        "unit_tests": "7 passed; detached-score gradient, smoothing control, empty fallback, batch/diagonal connectivity, ignore, exact state/inference and overlap head gradients",
        "operator_evidence": "outputs/overnight_unet_20261012/bicc_semantic_preparation/operator_audit.json"})
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
