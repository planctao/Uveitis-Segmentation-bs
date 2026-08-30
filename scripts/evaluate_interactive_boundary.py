"""Evaluate FA-UBIR oracle and deployment-safe policy click curves."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_interactive_boundary import build_loader, build_model, load_config, resolve_config  # noqa: E402
from bs.click_simulator import simulate_click_sequence  # noqa: E402
from bs.interactive_boundary import (  # noqa: E402
    boundary_f1_score,
    make_uncertainty_target,
    refine_boundary_with_clicks,
)
from bs.multilabel import PaperDice, masks_to_paper_targets  # noqa: E402
from bs.paths import project_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fa_ubir_v1.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", choices=["f1", "f2", "f3", "f4", "f5"], required=True)
    parser.add_argument("--clicks", default="0,1,3,5")
    parser.add_argument("--mode", choices=["policy", "oracle", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--roi-size", type=int, default=None)
    parser.add_argument("--stop-priority", type=float, default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _evaluate_one(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    config: dict[str, Any],
    click_count: int,
    policy: bool,
    roi_size: int | None,
    stop_priority: float | None,
) -> dict[str, float]:
    model.eval()
    metric = PaperDice(
        ignore_index=int(config.get("data", {}).get("ignore_index", 255)),
        threshold=config.get("metric", {}).get("threshold", 0.5),
    )
    boundary_total = 0.0
    uncertainty_mae = 0.0
    stopped_total = 0.0
    batches = 0
    click_cfg = config.get("clicks", {})
    for raw_batch in loader:
        batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in raw_batch.items()
        }
        target, valid = masks_to_paper_targets(
            batch["mask"], int(config.get("data", {}).get("ignore_index", 255))
        )
        kwargs: dict[str, Any] = {}
        if not policy:
            kwargs["positive_points"], kwargs["negative_points"] = simulate_click_sequence(
                target,
                torch.sigmoid(batch["dino_logits"]) >= float(click_cfg.get("threshold", 0.5)),
                num_clicks=int(click_count),
                strategy=str(click_cfg.get("oracle_strategy", "farthest")),
            )
        result = refine_boundary_with_clicks(
            model,
            batch["image"],
            batch["dino_logits"],
            int(click_count),
            threshold=float(click_cfg.get("threshold", 0.5)),
            radius=int(click_cfg.get("radius", 8)),
            mode=str(click_cfg.get("mode", "gaussian")),
            min_separation=int(click_cfg.get("min_separation", 16)),
            roi_size=(
                int(roi_size)
                if roi_size is not None
                else (int(config.get("refiner", {}).get("roi_size", 0)) or None)
            ),
            stop_priority=(
                float(stop_priority)
                if stop_priority is not None
                else config.get("clicks", {}).get("stop_priority")
            ),
            **kwargs,
        )
        metric.update(result.logits.detach().cpu(), batch["mask"].detach().cpu())
        boundary_total += float(boundary_f1_score(result.logits, target, valid).item())
        uncertainty_target = make_uncertainty_target(
            batch["dino_logits"], target, valid,
            boundary_kernel=int(config.get("boundary", {}).get("kernel", 5)),
        )
        uncertainty_mae += float((result.uncertainty - uncertainty_target).abs().mean().item())
        stopped_total += float(result.stopped.float().mean().item())
        batches += 1
    values = metric.compute()
    prefix = "policy" if policy else "oracle"
    denominator = max(1, batches)
    return {
        **{f"{prefix}_click{click_count}_{key}": float(value) for key, value in values.items()},
        f"{prefix}_click{click_count}_boundary_f1": boundary_total / denominator,
        f"{prefix}_click{click_count}_uncertainty_mae": uncertainty_mae / denominator,
        f"{prefix}_click{click_count}_stopped_fraction": stopped_total / denominator,
        f"{prefix}_click{click_count}_batches": float(batches),
    }


def main() -> None:
    args = parse_args()
    config = resolve_config(load_config(args.config), args)
    config.setdefault("train", {})["folds_to_run"] = [args.fold]
    checkpoint_path = project_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint.get("config")
    if isinstance(checkpoint_config, dict):
        # Keep model/loss architecture from the checkpoint, while allowing the
        # caller to select a different evaluation fold and output settings.
        merged = dict(checkpoint_config)
        for key, value in config.items():
            if key not in {"model", "loss"}:
                merged[key] = value
        config = merged
    device_name = str(config.get("runtime", {}).get("device", "cuda"))
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    loader = build_loader(config, args.fold, "val")
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    model.eval()
    click_counts = [int(item.strip()) for item in args.clicks.split(",") if item.strip()]
    if any(item < 0 for item in click_counts):
        raise ValueError("click counts must be non-negative")

    metrics: dict[str, float] = {}
    for count in click_counts:
        if args.mode in {"oracle", "both"}:
            metrics.update(_evaluate_one(model, loader, device, config, count, False, args.roi_size, args.stop_priority))
        if args.mode in {"policy", "both"}:
            metrics.update(_evaluate_one(model, loader, device, config, count, True, args.roi_size, args.stop_priority))
    result = {
        "fold": args.fold,
        "checkpoint": str(checkpoint_path),
        "clicks": click_counts,
        "mode": args.mode,
        "roi_size": args.roi_size,
        "stop_priority": args.stop_priority,
        "oracle_free_policy": True,
        "metrics": metrics,
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        output = project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
