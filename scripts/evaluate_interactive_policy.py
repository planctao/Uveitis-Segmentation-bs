"""Evaluate the deployment-safe uncertainty-guided click policy.

Unlike ``evaluate_interactive_refiner.py``, this script never uses the target
mask to choose clicks.  The mask is used only after inference to compute the
reported Dice, making the oracle-to-deployment gap measurable.
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

from train_interactive_refiner import build_loader, build_model, load_config, resolve_config, get_feature_builder  # noqa: E402
from bs.interactive_refiner import policy_refine_with_clicks  # noqa: E402
from bs.multilabel import PaperDice  # noqa: E402
from bs.paths import project_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dino_sam_refiner_uag.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", choices=["f1", "f2", "f3", "f4", "f5"], required=True)
    parser.add_argument("--clicks", default="0,1,3,5")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--min-separation", type=int, default=16)
    parser.add_argument(
        "--stop-priority",
        type=float,
        default=None,
        help="Optional automatic stop threshold on the click priority score.",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    config: dict,
    click_count: int,
    min_separation: int,
    stop_priority: float | None,
) -> dict[str, float]:
    metric = PaperDice(
        ignore_index=int(config.get("data", {}).get("ignore_index", 255)),
        threshold=config.get("metric", {}).get("threshold", 0.5),
    )
    total = 0.0
    batches = 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        dino_logits = batch["dino_logits"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        result = policy_refine_with_clicks(
            model,
            image,
            dino_logits,
            click_count,
            threshold=float(config.get("clicks", {}).get("threshold", 0.5)),
            radius=int(config.get("clicks", {}).get("radius", 10)),
            mode=str(config.get("clicks", {}).get("mode", "gaussian")),
            min_separation=min_separation,
            stop_priority=stop_priority,
            feature_builder=get_feature_builder(config),
            residual_step_limit=(
                float(config.get("clicks", {}).get("residual_step_limit"))
                if config.get("clicks", {}).get("residual_step_limit") not in (None, 0, 0.0)
                else None
            ),
            stop_gradient=bool(config.get("clicks", {}).get("stop_gradient", False)),
        )
        metric.update(result.logits.detach().cpu(), mask.detach().cpu())
        batches += 1
    values = metric.compute()
    return {f"policy_click{click_count}_{key}": float(value) for key, value in values.items()} | {
        f"policy_click{click_count}_batches": float(batches),
    }


def main() -> None:
    args = parse_args()
    config = resolve_config(load_config(args.config), args)
    config.setdefault("train", {})["folds_to_run"] = [args.fold]
    checkpoint_path = project_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_config = checkpoint.get("config")
    if isinstance(checkpoint_config, dict):
        checkpoint_config.update({k: v for k, v in config.items() if k not in {"model", "loss"}})
        config = checkpoint_config
    device_name = config.get("runtime", {}).get("device", "cuda")
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    loader = build_loader(config, args.fold, "val")
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    model.eval()

    counts = [int(item.strip()) for item in args.clicks.split(",") if item.strip()]
    if any(count < 0 for count in counts):
        raise ValueError("click counts must be non-negative")
    metrics: dict[str, float] = {}
    for count in counts:
        metrics.update(
            evaluate(
                model,
                loader,
                device,
                config,
                count,
                args.min_separation,
                args.stop_priority,
            )
        )
    result = {
        "fold": args.fold,
        "checkpoint": str(checkpoint_path),
        "clicks": counts,
        "min_separation": int(args.min_separation),
        "stop_priority": args.stop_priority,
        "metrics": metrics,
        "oracle_free": True,
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        output = project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
