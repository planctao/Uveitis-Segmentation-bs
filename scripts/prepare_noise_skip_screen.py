"""Prepare fixed NoiseUNet location controls as a lower-priority fallback."""

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
from scripts.train_noise_skip_multilabel import build_model


def main():
    torch.set_num_threads(2)
    overnight = PROJECT_ROOT / "outputs/overnight_unet_20261012"
    root = overnight / "noise_skip_screen"
    root.mkdir(exist_ok=True)
    if (root / "plan.json").exists() or (root / "queue_state.json").exists():
        raise RuntimeError("Do not overwrite an existing screen")
    reference_run = PROJECT_ROOT / "outputs/edg_validation_20261011/runs/paired_concat_s42_f1_30_256"
    original = json.loads((reference_run / "resolved_config.json").read_text())
    reference = json.loads((reference_run / "f1/initial_weight_fingerprints.json").read_text())
    jobs, audit = [], {}
    for mode in ["skip", "encoder"]:
        config = copy.deepcopy(original)
        name = f"noiseunet_{mode}_uniform_d0p5_s42_f1_30_256"
        config["project"]["name"] = name
        config["outputs"]["root"] = str(root.relative_to(PROJECT_ROOT) / "runs")
        config["model"]["noise_skip"] = {"mode": mode, "bound": .5, "training_only": True,
                                        "stages": "all five skip-producing encoder stages"}
        config["reproduction"] = {"experiment_class": "bounded_skip_feature_regularization_location_screen",
            "paper_url": "https://arxiv.org/abs/2606.04427v1", "no_public_author_code_found": True,
            "deviations": ["Original six-level FA nnUNet and original AFT+BCE instead of paperU-Net/BCE+Dice",
                "30epochs instead of400; original complete dataset/paired recipe retained",
                "Training-only bounded uniform.5; inference noise policy absent from paper, explicit clean adaptation",
                "Independent per-stage/epoch RNG for paired noise and epoch-boundary resume; no changes to data RNG"],
            "not_claimed": "No calibration, fuzzy-membership, worst-case guarantee or complete author-framework reproduction",
            "admission": "Fallback only after CAT complete replication does not support strong transfer, realCUDA optimizer probe and full two-arm window",
            "advance_rule": "Skip must beat same-seed plain and encoder-noise control; require additional complete matched seeds before formal strong transfer"}
        for section in ["data", "train", "loss", "metric", "augmentations", "runtime"]:
            if config[section] != original[section]:
                raise RuntimeError(f"Paired recipe changed: {section}")
        model = initialize_paired_model(build_model, config)
        if initial_weight_fingerprints(model) != reference:
            raise RuntimeError("All original initialization tensors must match")
        audit[name] = {"initial_original_tensors_matched": len(reference),
                       "parameters": sum(parameter.numel() for parameter in model.parameters()),
                       "no_new_parameters": True, "original_evaluation_factory_valid": True}
        path = root / "input_configs" / f"{name}.yaml"
        path.parent.mkdir(exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        jobs.append({"id": name, "config": str(path.relative_to(PROJECT_ROOT)), "fold": "f1",
            "training_script": "scripts/train_noise_skip_multilabel.py",
            "reference_diagnostics": "outputs/edg_validation_20261011/diagnostics/paired_concat_s42_f1_30_256"})
        del model
    atomic_json(root / "plan.json", {"jobs": jobs, "created_epoch": time.time(),
        "training_started_at_preparation": False, "expected_initial_difference_suffixes": [],
        "scope": "Independent fixed-bound location comparison with original clean-inference architecture; not combined with CAT/MIL"})
    manifest = json.loads((overnight / "component_screen/source_manifest.json").read_text())
    for name, expected in manifest["source_sha256"].items():
        if hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Original training source changed: {name}")
    sources = list(manifest["source_sha256"]) + ["src/bs/noise_skip_unet.py", "scripts/train_noise_skip_multilabel.py"]
    sources += [job["config"] for job in jobs] + [str((root / "plan.json").relative_to(PROJECT_ROOT))]
    atomic_json(root / "source_manifest.json", {"created_epoch": time.time(), "source_sha256": {
        name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() for name in sources}})
    report = {"status": "prepared_not_launched", "jobs": audit, "bound": .5,
        "bound_selected_from_paper_not_validation": True, "inference_noise_disabled": True,
        "CUDA_probe_passed": False, "scope": "Actual initialization/configuration preparation only; no accuracy score"}
    atomic_json(root / "preparation_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
