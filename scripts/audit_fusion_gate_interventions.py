"""Evaluate gate interventions on fixed trained weights and fixed thresholds.

This measures inference sensitivity. It does not replace retrained placement
ablations or estimate the benefit of training a different architecture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from scripts.fusion_evaluation_components import evaluation_components
from scripts.run_paper_reproduction import atomic_json
from scripts.train_unet_multilabel import set_seed, setup_logger
from bs.fusion_diagnostics import aggregate_errors, binary_error_record
from bs.multilabel import masks_to_paper_targets
from bs.recent_unet_methods import DifferenceGatedFusion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run, output = (PROJECT_ROOT / args.run).resolve(), (PROJECT_ROOT / args.output).resolve()
    output.relative_to(PROJECT_ROOT / "outputs")
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((run / "resolved_config.json").read_text())
    if config["model"]["name"] not in ("difference_fdg", "difference_edg"):
        raise ValueError("This audit uses U-Net fusion stages ordered deep to shallow")
    if config["train"].get("max_val_samples") or not config["data"].get("exclude_val_augmented"):
        raise ValueError("Require full clean validation")
    set_seed(config["project"]["seed"])
    build_loader, build_model, _, _ = evaluation_components(config)
    logger = setup_logger(output / "analysis.log", "gate_interventions")
    loader = build_loader(config, "f1", "val", logger)
    if len(loader.dataset) != 444:
        raise RuntimeError("Expected all 444 clean f1 images")
    checkpoint_path = run / "f1/checkpoints/best.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    best_epoch, expected = payload["epoch"], payload["best_metrics"]
    model = build_model(config)
    model.load_state_dict(payload["model"], strict=True)
    del payload
    device = torch.device(config["runtime"]["device"])
    model = model.to(device).eval()
    fusions = [m for m in model.modules() if isinstance(m, DifferenceGatedFusion)]
    if len(fusions) != 5 or not all(m.gating_enabled for m in fusions):
        raise ValueError("Require a trained model with all five fusion stages gated")
    thresholds = {"fixed_0.5": 0.5, "shared": float(expected["paper_sweep_best_threshold"])}
    interventions = {"trained_gates": (), "shallow_equal": (3, 4),
                     "deep_equal": (0, 1, 2), "all_equal": (0, 1, 2, 3, 4)}
    results = {}
    with torch.inference_mode():
        for name, replaced in interventions.items():
            for index, module in enumerate(fusions):
                module.gating_enabled = index not in replaced
            rows = []
            temporary = output / f"{name}.jsonl.tmp"
            with temporary.open("w") as stream:
                for batch in loader:
                    images = batch["image"].to(device, non_blocking=True)
                    masks = batch["mask"].to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=bool(config["train"]["amp"]) and device.type == "cuda"):
                        logits = model(images)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError(f"Nonfinite logits in {name}")
                    probabilities = logits.sigmoid()
                    targets, valid = masks_to_paper_targets(masks, config["data"]["ignore_index"])
                    targets = targets.cpu().numpy().astype(bool)
                    valid = valid[:, 0].cpu().numpy().astype(bool)
                    predictions = {key: (probabilities >= value).cpu().numpy() for key, value in thresholds.items()}
                    for index, sample_id in enumerate(batch["sample_id"]):
                        row = {"sample_id": sample_id, "protocols": {
                            protocol: {f"lesion_{lesion + 1}": binary_error_record(pred[index, lesion], targets[index, lesion], valid[index])
                                       for lesion in range(2)} for protocol, pred in predictions.items()}}
                        rows.append(row)
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
            if len(rows) != 444 or len({row["sample_id"] for row in rows}) != 444:
                raise RuntimeError(f"Incomplete or duplicate sample coverage for {name}")
            temporary.replace(output / f"{name}.jsonl")
            errors = {}
            for protocol in thresholds:
                errors[protocol] = {lesion: aggregate_errors([row["protocols"][protocol][lesion] for row in rows])
                                    for lesion in ("lesion_1", "lesion_2")}
                errors[protocol]["macro_global_dice"] = float(np.mean([errors[protocol][lesion]["global_pixel_dice"]
                                                                      for lesion in ("lesion_1", "lesion_2")]))
            if name == "trained_gates":
                for protocol, field in (("fixed_0.5", "paper_macro_dice"), ("shared", "paper_sweep_best_macro_dice")):
                    if not math.isclose(errors[protocol]["macro_global_dice"], float(expected[field]), rel_tol=1e-6, abs_tol=1e-6):
                        raise RuntimeError("Unmodified model failed checkpoint metric replay")
            results[name] = {"equal_weight_stages": list(replaced), "errors": errors,
                             "macro_delta_from_trained_gates": {
                                 protocol: errors[protocol]["macro_global_dice"] - results.get("trained_gates", {"errors": errors})["errors"][protocol]["macro_global_dice"]
                                 for protocol in thresholds}}
            logger.info("%s shared_macro=%.6f", name, errors["shared"]["macro_global_dice"])
    report = {"run": str(run), "best_epoch": best_epoch, "validation_samples": 444,
              "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
              "trained_gate_macro_replay_passed": True, "thresholds_unchanged": thresholds,
              "stage_order": "0 deep to 4 shallow", "results": results,
              "scope": "inference interventions with trained feature branches and BN statistics fixed; no retraining, threshold selection, training-effect estimate, or independent-test claim"}
    atomic_json(output / "summary.json", report)
    print(json.dumps({name: result["macro_delta_from_trained_gates"] for name, result in results.items()}, indent=2))


if __name__ == "__main__":
    main()
