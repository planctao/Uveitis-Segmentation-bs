"""Inspect actual AdamW step counters instead of assuming every AMP batch updated."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.fusion_evaluation_components import evaluation_components
from scripts.run_paper_reproduction import atomic_json
from scripts import train_unet_multilabel as unet
from bs.dataset import discover_samples
from bs.paths import project_path


def audit(run: Path, fold: str = "f1") -> dict:
    config = json.loads((run / "resolved_config.json").read_text())
    data, train = config["data"], config["train"]
    samples = discover_samples(dataset_root=project_path(data["root"]), folds=[f for f in data["folds"] if f != fold],
                               image_dir=data["image_dir"], mask_dir=data["mask_dir"],
                               hrnet_result_dir=data.get("hrnet_result_dir", "HRNet_Result"),
                               image_extensions=data["image_extensions"], mask_extensions=data["mask_extensions"],
                               result_extensions=data.get("result_extensions", data["image_extensions"]), exclude_augmented=False)
    sample_count = min(len(samples), int(train["max_train_samples"])) if train.get("max_train_samples") else len(samples)
    weighted = max(float(train.get("lesion1_sample_weight", 1) or 1), float(train.get("lesion2_sample_weight", 1) or 1)) > 1
    epoch_samples = max(1, round(sample_count * float(train.get("sampler_epoch_multiplier", 1) or 1))) if weighted else sample_count
    batches = epoch_samples // int(train["batch_size"])
    planned_per_epoch = math.ceil(batches / max(1, int(train.get("grad_accum_steps", 1))))
    _, factory, _, _ = evaluation_components(config)
    model = factory(config)
    if config["model"].get("backbone"):
        from scripts.train_dinov3_multilabel import build_optimizer
    else:
        build_optimizer = unet.build_optimizer
    optimizer = build_optimizer(model, config)
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    results = {}
    for label in ("best", "latest"):
        checkpoint = torch.load(run / fold / "checkpoints" / f"{label}.pt", map_location="cpu", weights_only=False)
        saved_optimizer = checkpoint["optimizer"]
        param_names = {}
        for group, saved_group in zip(optimizer.param_groups, saved_optimizer["param_groups"], strict=True):
            if len(group["params"]) != len(saved_group["params"]):
                raise RuntimeError("Optimizer parameter group differs from saved config")
            for parameter, index in zip(group["params"], saved_group["params"], strict=True):
                param_names[index] = names[id(parameter)]
        counters = {param_names[index]: int(state["step"].item()) for index, state in saved_optimizer["state"].items() if "step" in state}
        reference_name = next((name for name in counters if name.startswith("encoder.") or name.startswith("backbone.")), None)
        if reference_name is None:
            raise RuntimeError("No trained encoder parameter found in optimizer state")
        planned = int(checkpoint["epoch"]) * planned_per_epoch
        executed = counters[reference_name]
        if executed > planned:
            raise RuntimeError(f"Actual optimizer counter exceeds declared training budget: {executed} > {planned}")
        results[label] = {"epoch": checkpoint["epoch"], "planned_updates": planned,
                          "encoder_reference_parameter": reference_name, "executed_encoder_updates": executed,
                          "skipped_encoder_updates": planned - executed,
                          "distinct_parameter_step_counts": sorted(set(counters.values())),
                          "parameter_step_counts": counters}
        del checkpoint, saved_optimizer
    return {"run": str(run.resolve()), "fold": fold, "train_samples": sample_count, "epoch_sampler_samples": epoch_samples,
            "batch_size": train["batch_size"], "grad_accum_steps": train.get("grad_accum_steps", 1),
            "planned_updates_per_epoch": planned_per_epoch, "checkpoints": results,
            "scope": "actual AdamW counters for parameters with gradients; no-gradient branches have no optimizer state; counters since optimizer initialization"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--fold", default="f1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = audit(Path(args.run), args.fold)
    atomic_json(Path(args.output), report)
    print(json.dumps({"run": report["run"], "updates": {label: {key: value for key, value in row.items() if key != "parameter_step_counts"}
                                                         for label, row in report["checkpoints"].items()}}, indent=2))


if __name__ == "__main__":
    main()
