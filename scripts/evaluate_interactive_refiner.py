"""Evaluate a trained DINO-SAM interactive refiner at several click counts.

The script intentionally evaluates one fold at a time so that checkpoint/data
leakage is obvious.  For a five-fold report, invoke it once per fold (or use
the small shell loop shown in ``docs/dino_sam_refiner.md``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_interactive_refiner import (  # noqa: E402
    build_loader,
    build_loss,
    build_model,
    load_config,
    resolve_config,
    validate_clicks,
)
from bs.paths import project_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dino_sam_refiner.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", choices=["f1", "f2", "f3", "f4", "f5"], required=True)
    parser.add_argument("--clicks", default="0,1,3,5", help="Comma-separated click counts")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--iterative", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--residual-gate",
        default=None,
        choices=["none", "uncertainty", "uncertainty_click", "uncertainty_click_channelwise"],
    )
    parser.add_argument("--gate-floor", type=float, default=None)
    parser.add_argument("--gate-click-gain", type=float, default=None)
    parser.add_argument("--residual-scale", type=float, default=None)
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Reuse the training script's override semantics so evaluation sees the
    # exact same cache root, click policy and loss settings as training.
    config = load_config(args.config)
    config = resolve_config(config, args)
    config.setdefault("train", {})["folds_to_run"] = [args.fold]
    click_counts = [int(item.strip()) for item in args.clicks.split(",") if item.strip()]
    if any(item < 0 for item in click_counts):
        raise ValueError("click counts must be non-negative")

    checkpoint_path = project_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_config = checkpoint.get("config")
    if isinstance(checkpoint_config, dict):
        # The checkpoint owns model/loss/cache defaults; explicit CLI options
        # above still win for device/batch-size/workers/iterative evaluation.
        checkpoint_config.update({k: v for k, v in config.items() if k not in {"model", "loss"}})
        config = checkpoint_config
    # These gate switches are parameter-free and can be changed at evaluation
    # time without retraining the checkpoint.
    if args.residual_gate is not None:
        config.setdefault("model", {})["residual_gate"] = args.residual_gate
    if args.gate_floor is not None:
        config.setdefault("model", {})["residual_gate_floor"] = float(args.gate_floor)
    if args.gate_click_gain is not None:
        config.setdefault("model", {})["residual_gate_click_gain"] = float(args.gate_click_gain)
    if args.residual_scale is not None:
        if args.residual_scale < 0.0:
            raise ValueError("--residual-scale must be non-negative")
        config.setdefault("model", {})["residual_scale"] = float(args.residual_scale)
    device_name = config.get("runtime", {}).get("device", "cuda")
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    loader = build_loader(config, args.fold, "val")
    model = build_model(config).to(device)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    criterion = build_loss(config).to(device)
    model.eval()

    metrics: dict[str, float] = {}
    for count in click_counts:
        metrics.update(validate_clicks(model, loader, criterion, device, config, count))
    result = {
        "fold": args.fold,
        "checkpoint": str(checkpoint_path),
        "clicks": click_counts,
        "metrics": metrics,
        "iterative": bool(config.get("clicks", {}).get("iterative", False)),
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        output = project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
