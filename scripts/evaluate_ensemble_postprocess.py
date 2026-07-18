from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bs.adaptive_threshold import build_threshold_adapter
from bs.multilabel import PaperDice
from bs.fov import build_fov_masker
from bs.intensity_refine import build_intensity_refiner
from bs.paths import project_path
from bs.postprocess import build_postprocessor
from bs.tta import predict_with_tta


@dataclass(frozen=True)
class EnsembleMember:
    config_path: Path
    checkpoint_path: Path
    config: dict[str, Any]
    model: nn.Module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a probability/logit ensemble with TTA, morphology, and FOV postprocess.")
    parser.add_argument(
        "--member",
        action="append",
        required=True,
        help="One ensemble member as config:checkpoint. Repeat for multiple models.",
    )
    parser.add_argument("--fold", choices=["f1", "f2", "f3", "f4", "f5"], required=True)
    parser.add_argument(
        "--weights",
        default=None,
        help=(
            "Member weights like 0.7,0.3, or per-channel weights like "
            "0.8,0.2/0.6,0.4 for lesion_1 and lesion_2."
        ),
    )
    parser.add_argument("--average", choices=["prob", "logit"], default="prob")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--threshold", default=None, help="Scalar or two comma-separated thresholds, e.g. 0.5 or 0.5,0.9")
    parser.add_argument("--disable-tta", action="store_true")
    parser.add_argument("--disable-postprocess", action="store_true")
    parser.add_argument("--disable-intensity-refine", action="store_true")
    parser.add_argument("--disable-fov-mask", action="store_true")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with project_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_threshold(value: str | None, default: float | list[float]) -> float | list[float]:
    if value is None:
        return default
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts
    raise ValueError(f"Expected one or two thresholds, got {value}")


def parse_member(value: str) -> tuple[Path, Path]:
    if ":" not in value:
        raise ValueError(f"Expected --member config:checkpoint, got {value}")
    config_path, checkpoint_path = value.split(":", 1)
    return project_path(config_path), project_path(checkpoint_path)


def _normalize_weight_group(values: list[float], count: int) -> list[float]:
    if len(values) != count:
        raise ValueError(f"Expected {count} weights, got {len(values)}")
    total = sum(values)
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    return [weight / total for weight in values]


def parse_weights(value: str | None, count: int) -> list[float] | list[list[float]]:
    if value is None:
        return [1.0 / count for _ in range(count)]
    text = value.strip()
    separator = ";" if ";" in text else "/" if "/" in text else None
    if separator is not None:
        groups = [group.strip() for group in text.split(separator) if group.strip()]
        if not groups:
            raise ValueError("Expected at least one weight group")
        return [
            _normalize_weight_group([float(part.strip()) for part in group.split(",") if part.strip()], count)
            for group in groups
        ]
    weights = [float(part.strip()) for part in text.split(",") if part.strip()]
    return _normalize_weight_group(weights, count)


def validate_member_configs(configs: list[dict[str, Any]]) -> None:
    first = configs[0]
    keys = [
        ("train", "image_size"),
        ("data", "root"),
        ("data", "image_dir"),
        ("data", "mask_dir"),
        ("data", "ignore_index"),
        ("data", "label_values"),
    ]
    for config in configs[1:]:
        for section, key in keys:
            if config[section][key] != first[section][key]:
                raise ValueError(f"Ensemble members must share {section}.{key}: {config[section][key]} != {first[section][key]}")


def build_members(config_paths: list[Path], checkpoint_paths: list[Path], device: torch.device) -> list[EnsembleMember]:
    from scripts.evaluate_dinov3_postprocess import build_model, load_checkpoint

    members = []
    for config_path, checkpoint_path in zip(config_paths, checkpoint_paths):
        config = load_config(config_path)
        model = build_model(config).to(device)
        load_checkpoint(model, checkpoint_path)
        model.eval()
        members.append(
            EnsembleMember(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                config=config,
                model=model,
            )
        )
    return members


def apply_runtime_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["runtime"]["num_workers"] = args.num_workers
    config.setdefault("metric", {})
    config["metric"]["threshold"] = parse_threshold(args.threshold, config["metric"].get("threshold", 0.5))
    if args.disable_tta:
        config["metric"]["tta"] = {"enabled": False}
    if args.disable_postprocess:
        config["metric"]["postprocess"] = {"enabled": False}
    if args.disable_intensity_refine:
        config["metric"]["intensity_refine"] = {"enabled": False}
    if args.disable_fov_mask:
        config["metric"]["fov_mask"] = {"enabled": False}


def _ensemble_weight_tensor(
    weights: list[float] | list[list[float]],
    member_count: int,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    raw = torch.as_tensor(weights, device=device, dtype=dtype)
    if raw.ndim == 1:
        if raw.numel() != member_count:
            raise ValueError(f"Expected {member_count} member weights, got {raw.numel()}")
        total = raw.sum()
        if float(total.item()) <= 0.0:
            raise ValueError("weights must sum to a positive value")
        return (raw / total).view(member_count, 1, 1, 1)
    if raw.ndim == 2:
        if tuple(raw.shape) != (channels, member_count):
            raise ValueError(
                f"Per-channel weights must have shape [channels, members] = [{channels}, {member_count}], got {tuple(raw.shape)}"
            )
        totals = raw.sum(dim=1, keepdim=True)
        if bool((totals <= 0.0).any()):
            raise ValueError("each per-channel weight group must sum to a positive value")
        return (raw / totals).transpose(0, 1).contiguous().view(member_count, channels, 1, 1)
    raise ValueError(f"Unsupported weights shape: {tuple(raw.shape)}")


def _member_weight_summary(weights: list[float] | list[list[float]], member_idx: int) -> float | list[float]:
    if weights and isinstance(weights[0], list):
        return [float(channel_weights[member_idx]) for channel_weights in weights]
    return float(weights[member_idx])


def weighted_ensemble_logits(logits_list: list[torch.Tensor], weights: list[float] | list[list[float]], average: str) -> torch.Tensor:
    if not logits_list:
        raise ValueError("logits_list must not be empty")
    weight_tensor = _ensemble_weight_tensor(
        weights,
        member_count=len(logits_list),
        channels=logits_list[0].shape[1],
        device=logits_list[0].device,
        dtype=logits_list[0].dtype,
    )
    if average == "logit":
        result = torch.zeros_like(logits_list[0])
        for member_idx, logits in enumerate(logits_list):
            result = result + logits * weight_tensor[member_idx]
        return result

    probs = torch.zeros_like(logits_list[0])
    for member_idx, logits in enumerate(logits_list):
        probs = probs + torch.sigmoid(logits) * weight_tensor[member_idx]
    probs = probs.clamp(1e-6, 1.0 - 1e-6)
    return torch.logit(probs)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.evaluate_dinov3_postprocess import build_loader

    member_paths = [parse_member(value) for value in args.member]
    config_paths = [item[0] for item in member_paths]
    checkpoint_paths = [item[1] for item in member_paths]
    configs = [load_config(path) for path in config_paths]
    validate_member_configs(configs)

    base_config = configs[0]
    apply_runtime_overrides(base_config, args)
    device = torch.device(base_config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    loader = build_loader(base_config, args.fold)
    members = build_members(config_paths, checkpoint_paths, device)
    weights = parse_weights(args.weights, len(members))

    metric_cfg = base_config.get("metric", {})
    metric = PaperDice(
        ignore_index=int(base_config["data"]["ignore_index"]),
        threshold=metric_cfg["threshold"],
        postprocessor=build_postprocessor(metric_cfg.get("postprocess")),
        intensity_refiner=build_intensity_refiner(metric_cfg.get("intensity_refine")),
        fov_masker=build_fov_masker(metric_cfg.get("fov_mask")),
        threshold_adapter=build_threshold_adapter(metric_cfg.get("adaptive_threshold")),
    )
    tta_cfg = metric_cfg.get("tta", {"enabled": False})

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"ensemble {args.fold}", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            logits_list = []
            for member in members:
                with torch.amp.autocast("cuda", enabled=bool(member.config["train"].get("amp", True)) and device.type == "cuda"):
                    logits_list.append(predict_with_tta(member.model, images, tta_cfg))
            logits = weighted_ensemble_logits(logits_list, weights, args.average)
            metric.update(logits, masks, images)

    return {
        "fold": args.fold,
        "average": args.average,
        "weights": weights,
        "threshold": metric_cfg["threshold"],
        "tta": tta_cfg,
        "adaptive_threshold": metric_cfg.get("adaptive_threshold", {"enabled": False}),
        "postprocess": metric_cfg.get("postprocess", {"enabled": False}),
        "intensity_refine": metric_cfg.get("intensity_refine", {"enabled": False}),
        "fov_mask": metric_cfg.get("fov_mask", {"enabled": False}),
        "members": [
            {
                "config": str(member.config_path),
                "checkpoint": str(member.checkpoint_path),
                "backbone": member.config.get("model", {}).get("backbone", ""),
                "weight": _member_weight_summary(weights, idx),
            }
            for idx, member in enumerate(members)
        ],
        **metric.compute(),
    }


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        output_path = project_path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
