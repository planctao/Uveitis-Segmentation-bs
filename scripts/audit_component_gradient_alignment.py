"""Describe CAT/MIL gradient interactions at matched final checkpoints."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from bs.component_supervision import component_detection_terms
from bs.paired_experiments import seed_paired_epoch
from scripts import train_unet_multilabel as training
from scripts.run_paper_reproduction import atomic_json


def compare(a, b, names, prefix=None):
    selected = [(x, y) for name, x, y in zip(names, a, b, strict=True)
                if prefix is None or name.startswith(prefix)]
    aa = sum(float(x.double().square().sum()) for x, _ in selected if x is not None)
    bb = sum(float(y.double().square().sum()) for _, y in selected if y is not None)
    ab = sum(float((x.double() * y.double()).sum()) for x, y in selected if x is not None and y is not None)
    return {"main_norm": aa ** .5, "auxiliary_norm": bb ** .5,
            "auxiliary_to_main_norm": (bb / aa) ** .5 if aa else None,
            "cosine": ab / (aa * bb) ** .5 if aa and bb else None,
            "active_main_tensors": sum(x is not None for x, _ in selected),
            "active_auxiliary_tensors": sum(y is not None for _, y in selected)}


def main():
    torch.set_num_threads(2)
    overnight = PROJECT_ROOT / "outputs/overnight_unet_20261012"
    root = overnight / "component_screen"
    output = root / "matched_final_training_batch_gradient_alignment.json"
    if output.exists():
        raise RuntimeError("Keep the original descriptive gradient audit")
    baseline = PROJECT_ROOT / "outputs/edg_validation_20261011/runs/paired_concat_s42_f1_30_256"
    config = json.loads((baseline / "resolved_config.json").read_text())
    logger = logging.getLogger("component_gradient_alignment")
    loader = training.build_loader(config, "f1", "train", logger)
    val_loader = training.build_loader(config, "f1", "val", logger)
    seed_paired_epoch(config, 11, loader, val_loader)
    batch = next(iter(loader))
    if any(fold == "f1" for fold in batch["fold"]):
        raise RuntimeError("Held-out data in gradient audit")
    images, masks = batch["image"], batch["mask"]
    runs = {"plain": baseline, **{mode: root / "runs" / f"catmil_lesion2_{mode}_s42_f1_30_256"
                                for mode in ["cat", "mil", "joint"]}}
    results = {}
    for mode, run in runs.items():
        started = time.time()
        path = run / "f1/checkpoints/latest.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if int(checkpoint["epoch"]) != 30:
            raise RuntimeError("Final30epoch checkpoint required for every arm")
        # All four architectures/state keys are identical. Use the original
        # clean model, not any training-loss wrapper or checkpoint-selection loop.
        model = training.build_model(config).eval()
        model.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint
        criterion = training.build_loss(config)
        logits = model(images)
        base = criterion(logits, masks)
        terms = component_detection_terms(logits[:, 1].sigmoid(), (masks == 2) | (masks == 3),
                                          valid_mask=masks != config["data"]["ignore_index"])
        objectives = {"base": base, "CAT_weight0.1": .1 * terms["cat"], "MIL_weight0.2": .2 * terms["mil"]}
        names, parameters = zip(*[(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad])
        gradients = {name: torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
                     for name, value in objectives.items()}
        if not all(bool(torch.isfinite(g).all()) for values in gradients.values() for g in values if g is not None):
            raise FloatingPointError("Descriptive gradients nonfinite")
        joint = tuple((a if a is not None else 0) + (b if b is not None else 0)
                      if a is not None or b is not None else None
                      for a, b in zip(gradients["CAT_weight0.1"], gradients["MIL_weight0.2"], strict=True))
        pairs = {name: compare(gradients["base"], gradients[name], names)
                 for name in ["CAT_weight0.1", "MIL_weight0.2"]}
        pairs["joint_auxiliary"] = compare(gradients["base"], joint, names)
        cat_mil = compare(gradients["CAT_weight0.1"], gradients["MIL_weight0.2"], names)
        results[mode] = {"final_epoch": 30, "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "losses": {name: float(value.detach()) for name, value in objectives.items()},
            "L2_components": terms["component_count"], "all_parameter_comparisons": pairs,
            "CAT_vs_MIL": cat_mil,
            "joint_auxiliary_by_group": {prefix: compare(gradients["base"], joint, names, prefix)
                                         for prefix in ["encoder.", "decoder."]},
            "seconds": time.time() - started}
        atomic_json(output, {"status": "running", "results": results})
        print(mode, json.dumps(results[mode]["all_parameter_comparisons"]), flush=True)
        del model, criterion, logits, base, terms, objectives, gradients, joint, names, parameters
        gc.collect()
    report = {"status": "complete", "results": results, "device": "cpu", "precision": "FP32",
        "sample_ids": list(batch["sample_id"]), "folds": list(batch["fold"]), "paired_training_epoch": 11,
        "images_sha256": hashlib.sha256(images.contiguous().numpy().tobytes()).hexdigest(),
        "masks_sha256": hashlib.sha256(masks.contiguous().numpy().tobytes()).hexdigest(),
        "fixed_original_auxiliary_weights": {"cat": .1, "mil": .2},
        "scope": "One fixed paired training batch at four matched final30epoch states, eval-mode original model; full-parameter descriptive gradient alignment. No optimizer update, validation inference, weight tuning, or claim of full training-trajectory causality."}
    atomic_json(output, report)


if __name__ == "__main__":
    main()
