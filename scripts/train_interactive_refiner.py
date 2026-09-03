from __future__ import annotations

import argparse
import csv
import gc
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bs.click_simulator import (  # noqa: E402
    build_pseudo_sam_candidate,
    build_refiner_features,
    build_soft_prompt_features,
    simulate_click_heatmaps,
)
from bs.interactive_refiner import (  # noqa: E402
    InteractiveResidualRefiner,
    UncertaintyGatedResidualRefiner,
    click_consistency_loss,
    oracle_refine_with_target,
)
from bs.multilabel import AsymmetricFocalTverskyBCE, PaperDice, masks_to_paper_targets  # noqa: E402
from bs.paths import project_path  # noqa: E402


class CachedDinoDataset(Dataset):
    def __init__(self, manifest_path: Path, max_samples: int | None = None) -> None:
        self.rows: list[dict[str, str]] = []
        with manifest_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append(row)
        if not self.rows:
            raise RuntimeError(f"No cached samples found in {manifest_path}")
        if max_samples is not None:
            self.rows = self.rows[: max(0, int(max_samples))]
        if not self.rows:
            raise RuntimeError(f"No samples remain after max_samples={max_samples} in {manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        item = torch.load(row["path"], map_location="cpu", weights_only=True)
        image = item["image"]
        image_storage = str(item.get("image_storage", row.get("image_storage", "float16")))
        if image_storage == "uint8" or image.dtype == torch.uint8:
            # Cache stores compact RGB values; the DINO dataset uses the same
            # ImageNet normalization after optional appearance preprocessing.
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
            image = image.float().div(255.0)
            image = (image - mean) / std
        else:
            image = image.float()
        return {
            "sample_id": item["sample_id"],
            "fold": item["fold"],
            "image": image,
            "mask": item["mask"].long(),
            "dino_logits": item["dino_logits"].float(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train interactive residual refiner from cached DINOv3 predictions.")
    parser.add_argument("--config", default="configs/dino_sam_refiner.yaml")
    parser.add_argument("--fold", choices=["f1", "f2", "f3", "f4", "f5"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--project-name", default=None, help="Override project.name for a reproducible scan run.")
    parser.add_argument("--output-root", default=None, help="Override outputs.root for a scan suite.")
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional model checkpoint used to initialize a new fold before training.",
    )
    parser.add_argument("--residual-step-limit", type=float, default=None)
    parser.add_argument("--teacher-forcing-ratio", type=float, default=None)
    parser.add_argument(
        "--iterative",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Unroll the refiner after every click (true SAM-style interaction).",
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with project_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = {key: dict(value) if isinstance(value, dict) else value for key, value in config.items()}
    epochs = getattr(args, "epochs", None)
    batch_size = getattr(args, "batch_size", None)
    num_workers = getattr(args, "num_workers", None)
    learning_rate = getattr(args, "learning_rate", None)
    device = getattr(args, "device", None)
    max_train_samples = getattr(args, "max_train_samples", None)
    max_val_samples = getattr(args, "max_val_samples", None)
    project_name = getattr(args, "project_name", None)
    output_root = getattr(args, "output_root", None)
    init_checkpoint = getattr(args, "init_checkpoint", None)
    residual_step_limit = getattr(args, "residual_step_limit", None)
    teacher_forcing_ratio = getattr(args, "teacher_forcing_ratio", None)
    iterative = getattr(args, "iterative", None)
    fold = getattr(args, "fold", None)
    if epochs is not None:
        config["train"]["epochs"] = epochs
    if batch_size is not None:
        config["train"]["batch_size"] = batch_size
    if num_workers is not None:
        config["runtime"]["num_workers"] = num_workers
    if learning_rate is not None:
        config["train"]["learning_rate"] = learning_rate
    if device is not None:
        config["runtime"]["device"] = device
    if max_train_samples is not None:
        config.setdefault("train", {})["max_train_samples"] = max_train_samples
    if max_val_samples is not None:
        config.setdefault("train", {})["max_val_samples"] = max_val_samples
    if project_name is not None:
        config.setdefault("project", {})["name"] = project_name
    if output_root is not None:
        config.setdefault("outputs", {})["root"] = output_root
    if init_checkpoint is not None:
        config.setdefault("train", {})["init_checkpoint"] = init_checkpoint
    if residual_step_limit is not None:
        if residual_step_limit < 0.0:
            raise ValueError("--residual-step-limit must be non-negative")
        config.setdefault("clicks", {})["residual_step_limit"] = residual_step_limit
    if teacher_forcing_ratio is not None:
        if not 0.0 <= teacher_forcing_ratio <= 1.0:
            raise ValueError("--teacher-forcing-ratio must be in [0, 1]")
        config.setdefault("clicks", {})["teacher_forcing_ratio"] = teacher_forcing_ratio
    if iterative is not None:
        config.setdefault("clicks", {})["iterative"] = bool(iterative)
    if fold is not None:
        config["train"]["folds_to_run"] = [fold]
    return config


def setup_logger(path: Path, name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loader(config: dict[str, Any], fold: str, split: str) -> DataLoader:
    manifest = project_path(config["cache"]["root"]) / fold / f"{split}_manifest.csv"
    limit_key = "max_train_samples" if split == "train" else "max_val_samples"
    dataset = CachedDinoDataset(manifest, config.get("train", {}).get(limit_key))
    return DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=split == "train",
        num_workers=int(config["runtime"].get("num_workers", 4)),
        pin_memory=True,
        # Keep the final (and possibly only) batch.  This matters for the
        # small-fold smoke runs used to validate the pipeline and does not
        # affect the full five-fold experiment.
        drop_last=False,
        persistent_workers=int(config["runtime"].get("num_workers", 4)) > 0,
    )


def build_loss(config: dict[str, Any]) -> nn.Module:
    loss_cfg = config["loss"]
    return AsymmetricFocalTverskyBCE(
        pos_weight=loss_cfg.get("pos_weight", [3.0, 60.0]),
        bce_weight=float(loss_cfg.get("bce_weight", 0.5)),
        tversky_weight=float(loss_cfg.get("tversky_weight", 1.0)),
        alpha=float(loss_cfg.get("tversky_alpha", 0.2)),
        beta=float(loss_cfg.get("tversky_beta", 0.8)),
        gamma=float(loss_cfg.get("focal_gamma", 0.75)),
        ignore_index=int(config["data"].get("ignore_index", 255)),
        boundary_dice_weight=float(loss_cfg.get("boundary_dice_weight", 0.0)),
        boundary_dice_kernel=int(loss_cfg.get("boundary_dice_kernel", 5)),
    )


def build_model(config: dict[str, Any]) -> InteractiveResidualRefiner:
    model_cfg = config["model"]
    model_type = str(model_cfg.get("type", "legacy")).lower()
    model_cls = (
        UncertaintyGatedResidualRefiner
        if model_type in {"uncertainty_gated", "uag", "v2"}
        else InteractiveResidualRefiner
    )
    return model_cls(
        in_channels=int(model_cfg.get("in_channels", 13)),
        out_channels=int(model_cfg.get("out_channels", 2)),
        base_channels=int(model_cfg.get("base_channels", 32)),
        residual_scale=float(model_cfg.get("residual_scale", 1.0)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        residual_gate=str(model_cfg.get("residual_gate", "none")),
        residual_gate_floor=float(model_cfg.get("residual_gate_floor", 0.25)),
        residual_gate_click_gain=float(model_cfg.get("residual_gate_click_gain", 0.75)),
    )


def make_inputs(
    batch: dict[str, Tensor], config: dict[str, Any], num_clicks: int, *, return_prompts: bool = False
) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    image = batch["image"]
    mask = batch["mask"]
    dino_logits = batch["dino_logits"]
    target, _ = masks_to_paper_targets(mask, int(config["data"].get("ignore_index", 255)))
    probs = torch.sigmoid(dino_logits)
    click_cfg = config["clicks"]
    positive, negative = simulate_click_heatmaps(
        target,
        probs,
        num_clicks=num_clicks,
        threshold=float(click_cfg.get("threshold", 0.5)),
        radius=resolve_click_radius(click_cfg.get("radius", 8)),
        mode=str(click_cfg.get("mode", "disk")),
        strategy=str(click_cfg.get("strategy", "random")),
        cumulative=bool(click_cfg.get("cumulative", False)),
        click_jitter=float(click_cfg.get("jitter", 0.0) or 0.0),
        click_dropout=float(click_cfg.get("dropout", 0.0) or 0.0),
    )
    candidate = build_pseudo_sam_candidate(probs, positive, negative, threshold=float(click_cfg.get("threshold", 0.5)))
    builder = get_feature_builder(config)
    features = builder(image, dino_logits, candidate, positive, negative)
    if return_prompts:
        return features, dino_logits, mask, positive, negative
    return features, dino_logits, mask


def get_feature_builder(config: dict[str, Any]):
    mode = str(config.get("model", {}).get("feature_mode", "legacy")).lower()
    if mode in {"soft_signed", "uag", "v2"}:
        return build_soft_prompt_features
    return build_refiner_features


def resolve_click_radius(value: Any) -> int | list[int]:
    """Normalize scalar or per-lesion click radius from YAML values."""
    if isinstance(value, str):
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
        if not values:
            raise ValueError("click radius string must contain at least one integer")
        return values[0] if len(values) == 1 else values
    if isinstance(value, (list, tuple)):
        values = [int(item) for item in value]
        if not values:
            raise ValueError("click radius sequence must not be empty")
        return values[0] if len(values) == 1 else values
    return int(value)


def consistency_loss(final_logits: Tensor, dino_logits: Tensor, config: dict[str, Any]) -> Tensor:
    weight = float(config["loss"].get("consistency_weight", 0.0) or 0.0)
    if weight <= 0.0:
        return final_logits.new_zeros(())
    threshold = float(config["loss"].get("consistency_confidence", 0.4))
    dino_probs = torch.sigmoid(dino_logits).detach()
    confident = (dino_probs - 0.5).abs() > threshold
    if not bool(confident.any()):
        return final_logits.new_zeros(())
    return F.binary_cross_entropy_with_logits(final_logits[confident], dino_probs[confident]) * weight


def click_bce_loss(final_logits: Tensor, mask: Tensor, features: Tensor, config: dict[str, Any]) -> Tensor:
    weight = float(config["loss"].get("click_bce_weight", 0.0) or 0.0)
    if weight <= 0.0:
        return final_logits.new_zeros(())
    target, valid = masks_to_paper_targets(mask, int(config["data"].get("ignore_index", 255)))
    # Legacy features end with positive/negative maps.  UAG-v2 appends a
    # signed prompt and reliability maps, so recover the original click maps
    # by their stable positions in the 17-channel layout.
    if features.shape[1] >= 15:
        clicks = features[:, 9:13].amax(dim=1, keepdim=True)
    else:
        clicks = features[:, -4:].amax(dim=1, keepdim=True)
    clicks = clicks.clamp(0.0, 1.0)
    if not bool((clicks > 0).any()):
        return final_logits.new_zeros(())
    valid = valid.expand_as(target).float()
    pixel_weight = (1.0 + float(config["loss"].get("click_area_gain", 4.0)) * clicks).expand_as(target) * valid
    bce = F.binary_cross_entropy_with_logits(final_logits, target.to(final_logits.device), reduction="none")
    return (bce * pixel_weight.to(final_logits.device)).sum() / pixel_weight.to(final_logits.device).sum().clamp_min(1.0) * weight


def rollout_options(config: dict[str, Any], *, training: bool) -> dict[str, Any]:
    """Resolve iterative-stability options without leaking teacher targets to eval."""
    click_cfg = config.get("clicks", {})
    raw_limit = click_cfg.get("residual_step_limit", None)
    limit = None if raw_limit is None or float(raw_limit) <= 0.0 else float(raw_limit)
    raw_total_limit = click_cfg.get("total_residual_limit", None)
    total_limit = None if raw_total_limit is None or float(raw_total_limit) <= 0.0 else float(raw_total_limit)
    ratio_key = "teacher_forcing_ratio" if training else "eval_teacher_forcing_ratio"
    return {
        "residual_step_limit": limit,
        "total_residual_limit": total_limit,
        "stop_gradient": bool(click_cfg.get("stop_gradient", False)),
        "teacher_forcing_ratio": float(click_cfg.get(ratio_key, 0.0) or 0.0),
        "teacher_forcing_logit_scale": float(click_cfg.get("teacher_forcing_logit_scale", 4.0)),
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: dict[str, Any],
    epoch: int,
    logger: logging.Logger,
) -> dict[str, float]:
    model.train()
    total = 0.0
    click_choices = [int(v) for v in config["clicks"].get("train_clicks", [0, 1, 3, 5])]
    progress = tqdm(loader, desc=f"train {epoch}", leave=False)
    for step, batch in enumerate(progress, start=1):
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        num_clicks = random.choice(click_choices)
        dino_logits = batch["dino_logits"]
        mask = batch["mask"]
        with torch.autocast(device_type=device.type, enabled=bool(config["runtime"].get("amp", True)) and device.type == "cuda"):
            if bool(config.get("clicks", {}).get("iterative", False)):
                target, _ = masks_to_paper_targets(
                    mask, int(config["data"].get("ignore_index", 255))
                )
                result = oracle_refine_with_target(
                    model,
                    batch["image"],
                    dino_logits,
                    target,
                    num_clicks,
                    threshold=float(config["clicks"].get("threshold", 0.5)),
                    radius=resolve_click_radius(config["clicks"].get("radius", 8)),
                    mode=str(config["clicks"].get("mode", "disk")),
                    strategy=str(config["clicks"].get("strategy", "random")),
                    feature_builder=get_feature_builder(config),
                    click_jitter=float(config["clicks"].get("jitter", 0.0) or 0.0),
                    click_dropout=float(config["clicks"].get("dropout", 0.0) or 0.0),
                    **rollout_options(config, training=True),
                )
                final_logits = result.logits
                # Reuse the final cumulative click maps for the auxiliary
                # click losses.  The legacy MVP used ``None`` here, which
                # silently disabled click supervision in strict iterative
                # training; UAG keeps the closed loop while supervising the
                # actual prompts that produced the final prediction.
                positive_clicks = result.positive_clicks
                negative_clicks = result.negative_clicks
                if positive_clicks is None or negative_clicks is None:
                    features = None
                else:
                    candidate = build_pseudo_sam_candidate(
                        torch.sigmoid(final_logits),
                        positive_clicks,
                        negative_clicks,
                        threshold=float(config["clicks"].get("threshold", 0.5)),
                    )
                    features = get_feature_builder(config)(
                        batch["image"],
                        final_logits.detach(),
                        candidate,
                        positive_clicks,
                        negative_clicks,
                    )
            else:
                features, dino_logits, mask = make_inputs(batch, config, num_clicks)
                final_logits = model(features, dino_logits)
            loss = criterion(final_logits, mask)
            loss = loss + consistency_loss(final_logits, dino_logits, config)
            if features is not None:
                loss = loss + click_bce_loss(final_logits, mask, features, config)
                click_weight = float(config["loss"].get("click_consistency_weight", 0.0) or 0.0)
                if click_weight > 0.0:
                    loss = loss + click_weight * click_consistency_loss(
                        final_logits,
                        features[:, 9:11] if features.shape[1] >= 15 else features[:, -4:-2],
                        features[:, 11:13] if features.shape[1] >= 15 else features[:, -2:],
                        margin=float(config["loss"].get("click_logit_margin", 2.0)),
                    )
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if float(config["train"].get("clip_grad_norm", 0.0) or 0.0) > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["clip_grad_norm"]))
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach().item())
        progress.set_postfix(loss=f"{total / step:.4f}", clicks=num_clicks)
    result = {"loss": total / max(len(loader), 1)}
    logger.info("train epoch=%d loss=%.6f", epoch, result["loss"])
    return result


@torch.no_grad()
def validate_clicks(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: dict[str, Any],
    num_clicks: int,
) -> dict[str, float]:
    model.eval()
    metrics = PaperDice(ignore_index=int(config["data"].get("ignore_index", 255)), threshold=config["metric"].get("threshold", 0.5))
    total = 0.0
    for batch in tqdm(loader, desc=f"val {num_clicks} clicks", leave=False):
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        mask = batch["mask"]
        dino_logits = batch["dino_logits"]
        if bool(config.get("clicks", {}).get("iterative", False)):
            target, _ = masks_to_paper_targets(
                mask, int(config["data"].get("ignore_index", 255))
            )
            result = oracle_refine_with_target(
                model,
                batch["image"],
                dino_logits,
                target,
                num_clicks,
                threshold=float(config["clicks"].get("threshold", 0.5)),
                radius=resolve_click_radius(config["clicks"].get("radius", 8)),
                mode=str(config["clicks"].get("mode", "disk")),
                strategy=str(config["clicks"].get("strategy", "random")),
                feature_builder=get_feature_builder(config),
                click_jitter=0.0,
                click_dropout=0.0,
                **rollout_options(config, training=False),
            )
            final_logits = result.logits
        else:
            features, dino_logits, mask = make_inputs(batch, config, num_clicks)
            final_logits = model(features, dino_logits)
        loss = criterion(final_logits, mask)
        total += float(loss.detach().item())
        metrics.update(final_logits.detach().cpu(), mask.detach().cpu())
    result = {f"click{num_clicks}_loss": total / max(len(loader), 1)}
    for key, value in metrics.compute().items():
        result[f"click{num_clicks}_{key}"] = value
    return result


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_score: float,
    config: dict[str, Any],
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    state: dict[str, Any] = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_score": best_score,
        "config": config,
    }
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    torch.save(state, path)


def load_model_initialization(model: nn.Module, checkpoint_path: str | Path, logger: logging.Logger) -> None:
    """Load an MVP checkpoint, allowing only input-channel expansion.

    The soft-signed prompt variant adds four parameter-free channels to the
    original 13-channel MVP input.  Copying the old stem weights and zeroing
    the new channels makes the expanded model exactly equivalent to MVP at
    initialization, while keeping all deeper weights and optimizer behavior
    trainable.  Any other shape mismatch remains an error so experiments do
    not silently start from a partially unrelated model.
    """
    init_path = project_path(str(checkpoint_path))
    if not init_path.is_file():
        raise FileNotFoundError(f"initialization checkpoint not found: {init_path}")
    init_state = torch.load(init_path, map_location="cpu", weights_only=False)
    source = init_state.get("model", init_state)
    target = model.state_dict()
    if not isinstance(source, dict):
        raise TypeError(f"checkpoint model state must be a mapping, got {type(source)!r}")
    if set(source) != set(target):
        missing = sorted(set(target) - set(source))
        extra = sorted(set(source) - set(target))
        raise RuntimeError(f"checkpoint keys differ; missing={missing[:5]}, extra={extra[:5]}")
    copied = 0
    expanded = 0
    for key, target_value in target.items():
        source_value = source[key]
        if source_value.shape == target_value.shape:
            target[key].copy_(source_value)
            copied += 1
            continue
        # The only supported expansion is the first convolution's input
        # channel dimension (13 -> 17 for signed/reliability prompts).
        if (
            source_value.ndim == 4
            and target_value.ndim == 4
            and source_value.shape[0] == target_value.shape[0]
            and source_value.shape[2:] == target_value.shape[2:]
            and source_value.shape[1] < target_value.shape[1]
            and key.endswith("stem.0.0.weight")
        ):
            target[key].zero_()
            target[key][:, : source_value.shape[1]].copy_(source_value)
            expanded += 1
            continue
        raise RuntimeError(
            f"unsupported checkpoint shape mismatch for {key}: "
            f"source={tuple(source_value.shape)} target={tuple(target_value.shape)}"
        )
    model.load_state_dict(target, strict=True)
    logger.info(
        "initialized model from %s (copied=%d tensors, expanded_input_stem=%d)",
        init_path,
        copied,
        expanded,
    )


def configure_prompt_adapter_only(
    model: nn.Module,
    *,
    original_input_channels: int = 13,
    train_output_projection: bool = True,
) -> None:
    """Freeze an MVP-expanded refiner and train only appended input channels.

    This is useful for a strict incremental ablation: all original MVP
    parameters remain fixed, while the first convolution and residual output
    projection learn how the newly appended prompt channels should modulate
    the frozen decoder.  The gradient mask prevents accidental updates to the
    original 13 channels.
    """
    if not hasattr(model, "stem") or not hasattr(model.stem[0], "__getitem__"):
        raise ValueError("prompt adapter requires a refiner with a convolutional stem")
    first_conv = model.stem[0][0]
    if not isinstance(first_conv, nn.Conv2d) or first_conv.in_channels <= original_input_channels:
        raise ValueError(
            "prompt adapter requires a stem Conv2d with appended input channels; "
            f"got {type(first_conv).__name__} in_channels={getattr(first_conv, 'in_channels', None)}"
        )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    first_conv.weight.requires_grad_(True)
    # The frozen decoder still propagates gradients to the appended stem
    # channels, so the output projection is optional.  Keeping it frozen is a
    # useful stricter variant: when all prompt channels are zero (0 clicks),
    # the initialized model remains exactly equal to the MVP checkpoint.
    if train_output_projection:
        if not hasattr(model, "delta_head"):
            raise ValueError("prompt adapter requires a refiner with delta_head")
        for parameter in model.delta_head.parameters():
            parameter.requires_grad_(True)
    mask = torch.zeros_like(first_conv.weight)
    mask[:, int(original_input_channels) :] = 1.0
    first_conv.weight.register_hook(lambda gradient: gradient * mask)


def append_metrics(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(row.keys())
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train_fold(config: dict[str, Any], fold: str, root_dir: Path) -> dict[str, Any]:
    fold_dir = root_dir / fold
    (fold_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    logger = setup_logger(fold_dir / "train.log", f"interactive_refiner.{fold}.{time.time_ns()}")
    device = torch.device(config["runtime"].get("device", "cuda") if torch.cuda.is_available() else "cpu")
    epochs = int(config["train"]["epochs"])
    metrics_path = fold_dir / "metrics.csv"
    existing_rows: list[dict[str, str]] = []
    if metrics_path.exists():
        try:
            with metrics_path.open("r", encoding="utf-8", newline="") as handle:
                existing_rows = list(csv.DictReader(handle))
        except (OSError, csv.Error) as exc:
            logger.warning("could not read existing metrics for fold=%s: %s", fold, exc)
    existing_epochs = [int(row["epoch"]) for row in existing_rows if row.get("epoch", "").isdigit()]
    last_existing_epoch = max(existing_epochs, default=0)
    score_key = f"click{int(config['metric'].get('select_clicks', 3))}_paper_macro_dice"
    completed_scores = [float(row[score_key]) for row in existing_rows if row.get(score_key)]
    completed_best_score = max(completed_scores, default=-math.inf)
    completed_best_epoch = next(
        (int(row["epoch"]) for row in existing_rows if row.get(score_key) and float(row[score_key]) == completed_best_score),
        0,
    )
    latest_path = fold_dir / "checkpoints" / "latest.pt"
    best_path = fold_dir / "checkpoints" / "best.pt"
    if last_existing_epoch >= epochs and latest_path.is_file() and best_path.is_file():
        logger.info("fold=%s already complete through epoch=%d; reusing existing checkpoints", fold, last_existing_epoch)
        return {
            "fold": fold,
            "best_epoch": completed_best_epoch,
            "best_score": completed_best_score,
            "run_dir": str(fold_dir),
        }

    writer = SummaryWriter(str(fold_dir / "tensorboard"))
    train_loader: DataLoader | None = None
    val_loader: DataLoader | None = None
    model: nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR | None = None
    try:
        train_loader = build_loader(config, fold, "train")
        val_loader = build_loader(config, fold, "val")
        model = build_model(config).to(device)
        init_checkpoint = config.get("train", {}).get("init_checkpoint")
        if init_checkpoint and not latest_path.is_file():
            load_model_initialization(model, str(init_checkpoint), logger)
        if bool(config.get("train", {}).get("prompt_adapter_only", False)):
            configure_prompt_adapter_only(
                model,
                original_input_channels=int(config.get("train", {}).get("original_input_channels", 13)),
                train_output_projection=bool(
                    config.get("train", {}).get("prompt_adapter_train_output_projection", True)
                ),
            )
            logger.info("enabled prompt_adapter_only training")
        criterion = build_loss(config).to(device)
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=float(config["train"]["learning_rate"]),
            weight_decay=float(config["train"].get("weight_decay", 1e-4)),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
            eta_min=float(config["train"].get("min_learning_rate", 1e-6)),
        )
        scaler = torch.amp.GradScaler("cuda", enabled=bool(config["runtime"].get("amp", True)) and device.type == "cuda")
        best_score = completed_best_score
        best_epoch = completed_best_epoch
        start_epoch = last_existing_epoch + 1
        if latest_path.is_file():
            try:
                checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint["model"])
                if checkpoint.get("optimizer"):
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if scheduler is not None and checkpoint.get("scheduler"):
                    scheduler.load_state_dict(checkpoint["scheduler"])
                checkpoint_epoch = int(checkpoint.get("epoch", 0))
                start_epoch = max(start_epoch, checkpoint_epoch + 1)
                best_score = max(best_score, float(checkpoint.get("best_score", -math.inf)))
                logger.info("fold=%s resumed from checkpoint epoch=%d", fold, checkpoint_epoch)
            except (OSError, RuntimeError, KeyError, ValueError, TypeError) as exc:
                logger.warning("fold=%s checkpoint resume skipped: %s", fold, exc)
        if start_epoch > epochs:
            logger.info("fold=%s has no remaining epochs; preserving existing metrics", fold)
            return {
                "fold": fold,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "run_dir": str(fold_dir),
            }
        assert model is not None and optimizer is not None and scheduler is not None
        assert train_loader is not None and val_loader is not None
        eval_clicks = [int(v) for v in config["clicks"].get("eval_clicks", [0, 1, 3, 5])]
        logger.info("fold=%s train_samples=%d val_samples=%d start_epoch=%d", fold, len(train_loader.dataset), len(val_loader.dataset), start_epoch)
        for epoch in range(start_epoch, epochs + 1):
            train = train_epoch(model, train_loader, criterion, optimizer, scaler, device, config, epoch, logger)
            row: dict[str, Any] = {"epoch": epoch, "train_loss": train["loss"], "lr": optimizer.param_groups[0]["lr"]}
            for clicks in eval_clicks:
                row.update(validate_clicks(model, val_loader, criterion, device, config, clicks))
            scheduler.step()
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(key, value, epoch)
            append_metrics(metrics_path, row)
            score = float(row.get(score_key, row.get("click3_paper_macro_dice", -math.inf)))
            if score > best_score:
                best_score = score
                best_epoch = epoch
                save_checkpoint(best_path, model, optimizer, epoch, best_score, config, scheduler)
            save_checkpoint(latest_path, model, optimizer, epoch, best_score, config, scheduler)
            logger.info("epoch=%d best_epoch=%d best_score=%.6f row=%s", epoch, best_epoch, best_score, row)
        return {"fold": fold, "best_epoch": best_epoch, "best_score": best_score, "run_dir": str(fold_dir)}
    finally:
        writer.close()
        # Persistent DataLoader workers otherwise survive fold transitions and
        # can retain host memory/file descriptors for the next fold.
        del train_loader, val_loader, model, optimizer, scheduler
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = resolve_config(load_config(args.config), args)
    seed_everything(int(config["project"].get("seed", 42)))
    root_dir = project_path(config["outputs"].get("root", "outputs/interactive_refiner_runs")) / str(config["project"]["name"])
    root_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in config["train"].get("folds_to_run", ["f1", "f2", "f3", "f4", "f5"]):
        rows.append(train_fold(config, fold, root_dir))
        write_summary(root_dir / "fold_summary.csv", rows)


if __name__ == "__main__":
    main()
