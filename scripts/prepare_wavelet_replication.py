"""Prepare two additional paired Haar/spatial-phase seeds, without launching."""

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
from scripts.train_wavelet_decoder_multilabel import build_model
from scripts.run_paper_reproduction import atomic_json


def main():
    torch.set_num_threads(2)
    root = PROJECT_ROOT / "outputs/overnight_unet_20261012/wavelet_replication"
    root.mkdir(parents=True, exist_ok=True)
    if (root / "queue_state.json").exists():
        raise RuntimeError("Cannot change a launched plan")
    screen = root.parent / "wavelet_screen"
    jobs, audit = [], {}
    for seed in [2026, 3407]:
        baseline = PROJECT_ROOT / f"outputs/edg_validation_20261011/runs/paired_concat_s{seed}_f1_30_256"
        baseline_config = json.loads((baseline / "resolved_config.json").read_text())
        baseline_initial = json.loads((baseline / "f1/initial_weight_fingerprints.json").read_text())
        initial = {}
        for arm in ["haar", "spatial_phase"]:
            original = yaml.safe_load((screen / f"input_configs/acm_mswt_{arm}_s42_f1_30_256.yaml").read_text())
            config = copy.deepcopy(original)
            name = f"acm_mswt_{arm}_s{seed}_f1_30_256"
            config["project"].update(name=name, seed=seed)
            config["runtime"].update(paired_model_seed=seed, paired_data_seed=seed)
            config["outputs"]["root"] = str(root.relative_to(PROJECT_ROOT) / "runs")
            config["reproduction"]["training_protocol"] = f"seed{seed} f1 full1965 train444 clean val; unchanged30epoch batch8 recipe; paired common initial tensors and epoch data streams"
            for section in ["data", "train", "loss", "metric", "augmentations", "runtime"]:
                if config[section] != baseline_config[section]:
                    raise RuntimeError(f"Seed{seed} mismatched baseline recipe: {section}")
            model = initialize_paired_model(build_model, config)
            state = initial_weight_fingerprints(model)
            assert all(state[key] == value for key, value in baseline_initial.items())
            initial[arm] = state
            audit[name] = {"baseline_common_tensors_matched": len(baseline_initial),
                           "parameters": sum(p.numel() for p in model.parameters())}
            path = root / "input_configs" / f"{name}.yaml"
            path.parent.mkdir(exist_ok=True)
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            jobs.append({"id": name, "config": str(path.relative_to(PROJECT_ROOT)), "fold": "f1",
                "training_script": "scripts/train_wavelet_decoder_multilabel.py", "comparison_group": str(seed),
                "reference_diagnostics": f"outputs/edg_validation_20261011/diagnostics/paired_concat_s{seed}_f1_30_256"})
            del model
        differences = [key for key, value in initial["haar"].items() if value != initial["spatial_phase"][key]]
        assert len(differences) == 18 and all(key.endswith(".filters") for key in differences)
        audit[f"seed{seed}_fixed_basis_differences"] = differences
    atomic_json(root / "plan.json", {"jobs": jobs, "created_epoch": time.time(),
        "advance_rule": "Haar must exceed its corresponding plain and spatial-phase shared Macro in all three seeds before formal strong transfer; retain negative results and inspect both lesions, fixed Dice, FP/FN and tiny coverage.",
        "launch_rule": "Requires complete positive seed42 screen and replays; waits for the512 baseline training queue to free memory; minimum4 hours remaining for all four30epoch jobs.",
        "expected_initial_difference_suffixes": [".filters"], "training_started_at_preparation": False})
    manifest = json.loads((screen / "source_manifest.json").read_text())
    for name, expected in manifest["source_sha256"].items():
        assert hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() == expected
    atomic_json(root / "source_manifest.json", manifest)
    atomic_json(root / "preparation_audit.json", {"status": "prepared_not_launched", "jobs": audit,
                                                "all_trainable_initial_tensors_identical_by_seed": True})
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
