"""Train the FA-UBIR boundary/uncertainty interactive refiner.

The script consumes the compact cached DINO predictions used by the previous
interactive experiment.  Ground truth is used only to simulate training
clicks and to supervise the auxiliary boundary/error heads; deployment policy
evaluation is implemented separately in ``evaluate_interactive_boundary.py``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import logging
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_interactive_refiner import CachedDinoDataset  # noqa: E402
from bs.click_simulator import simulate_click_heatmaps, simulate_click_sequence  # noqa: E402
from bs.interactive_boundary import (  # noqa: E402
    BoundaryAwareInteractiveRefiner,
    boundary_f1_score,
    boundary_uncertainty_loss,
    make_uncertainty_target,
    refine_boundary_with_clicks,
)
from bs.interactive_refiner import click_consistency_loss  # noqa: E402
from bs.multilabel import AsymmetricFocalTverskyBCE, PaperDice, masks_to_paper_targets  # noqa: E402
from bs.paths import project_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fa_ubir_v1.yaml")
    parser.add_argument("--fold", choices=["f1", "f2", "f3", "f4", "f5"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--project-name", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with project_path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return value


def resolve_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = copy.deepcopy(config)
    overrides = {
        ("train", "epochs"): getattr(args, "epochs", None),
        ("train", "batch_size"): getattr(args, "batch_size", None),
        ("runtime", "num_workers"): getattr(args, "num_workers", None),
        ("train", "learning_rate"): getattr(args, "learning_rate", None),
        ("runtime", "device"): getattr(args, "device", None),
        ("train", "max_train_samples"): getattr(args, "max_train_samples", None),
        ("train", "max_val_samples"): getattr(args, "max_val_samples", None),
        ("project", "name"): getattr(args, "project_name", None),
        ("project", "seed"): getattr(args, "seed", None),
    }
    for (section, key), value in overrides.items():
        if value is not None:
            config.setdefault(section, {})[key] = value
    output_root = getattr(args, "output_root", None)
    fold = getattr(args, "fold", None)
    if output_root is not None:
        config.setdefault("outputs", {})["root"] = output_root
    if fold is not None:
        config.setdefault("train", {})["folds_to_run"] = [fold]
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(path: Path, name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def build_loader(config: dict[str, Any], fold: str, split: str) -> DataLoader:
    manifest = project_path(config["cache"]["root"]) / fold / f"{split}_manifest.csv"
    limit_key = "max_train_samples" if split == "train" else "max_val_samples"
    dataset = CachedDinoDataset(manifest, config.get("train", {}).get(limit_key))
    num_workers = int(config.get("runtime", {}).get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(config["train"].get("batch_size", 1)),
        shuffle=split == "train",
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


def build_model(config: dict[str, Any]) -> BoundaryAwareInteractiveRefiner:
    model_cfg = config.get("model", {})
    return BoundaryAwareInteractiveRefiner(
        in_channels=int(model_cfg.get("in_channels", 19)),
        out_channels=int(model_cfg.get("out_channels", 2)),
        image_channels=int(model_cfg.get("image_channels", 3)),
        base_channels=int(model_cfg.get("base_channels", 16)),
        residual_scale=float(model_cfg.get("residual_scale", 1.0)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        require_click_for_residual=bool(model_cfg.get("require_click_for_residual", True)),
        fa_aware_input=bool(model_cfg.get("fa_aware_input", True)),
        fa_aware_kernel=int(model_cfg.get("fa_aware_kernel", 5)),
    )


def build_segmentation_loss(config: dict[str, Any]) -> nn.Module:
    loss_cfg = config.get("loss", {})
    return AsymmetricFocalTverskyBCE(
        pos_weight=loss_cfg.get("pos_weight", [3.0, 60.0]),
        bce_weight=float(loss_cfg.get("bce_weight", 0.5)),
        tversky_weight=float(loss_cfg.get("tversky_weight", 1.0)),
        alpha=float(loss_cfg.get("tversky_alpha", 0.2)),
        beta=float(loss_cfg.get("tversky_beta", 0.8)),
        gamma=float(loss_cfg.get("focal_gamma", 0.75)),
        ignore_index=int(config.get("data", {}).get("ignore_index", 255)),
    )


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _click_maps(
    batch: dict[str, Tensor], config: dict[str, Any], num_clicks: int
) -> tuple[Tensor, Tensor]:
    target, _ = masks_to_paper_targets(batch["mask"], int(config.get("data", {}).get("ignore_index", 255)))
    click_cfg = config.get("clicks", {})
    return simulate_click_heatmaps(
        target,
        torch.sigmoid(batch["dino_logits"]),
        num_clicks=int(num_clicks),
        threshold=float(click_cfg.get("threshold", 0.5)),
        radius=int(click_cfg.get("radius", 8)),
        mode=str(click_cfg.get("mode", "gaussian")),
        strategy=str(click_cfg.get("strategy", "farthest")),
        cumulative=bool(click_cfg.get("cumulative", False)),
        click_jitter=float(click_cfg.get("jitter", 2.0) or 0.0),
        click_dropout=float(click_cfg.get("dropout", 0.1) or 0.0),
    )


def train_epoch(
    model: BoundaryAwareInteractiveRefiner,
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
    click_choices = [int(item) for item in config.get("clicks", {}).get("train_clicks", [0, 1, 3])]
    if any(item < 0 for item in click_choices):
        raise ValueError("clicks.train_clicks must be non-negative")
    amp_enabled = bool(config.get("runtime", {}).get("amp", True)) and device.type == "cuda"
    loss_cfg = config.get("loss", {})
    total = {"loss": 0.0, "seg_loss": 0.0, "aux_loss": 0.0}
    progress = tqdm(loader, desc=f"train {epoch}", leave=False)
    for step, raw_batch in enumerate(progress, start=1):
        batch = _move_batch(raw_batch, device)
        num_clicks = random.choice(click_choices)
        target, valid = masks_to_paper_targets(
            batch["mask"], int(config.get("data", {}).get("ignore_index", 255))
        )
        positive, negative = _click_maps(batch, config, num_clicks)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            aux = model.forward_with_aux(
                batch["image"],
                batch["dino_logits"],
                positive,
                negative,
                threshold=float(config.get("clicks", {}).get("threshold", 0.5)),
                entropy_weight=float(config.get("uncertainty", {}).get("entropy_weight", 0.5)),
                learned_weight=float(config.get("uncertainty", {}).get("learned_weight", 0.3)),
                boundary_conflict_weight=float(config.get("uncertainty", {}).get("boundary_conflict_weight", 0.2)),
            )
            seg_loss = criterion(aux["logits"], batch["mask"])
            aux_loss, aux_values = boundary_uncertainty_loss(
                aux["boundary_logits"],
                aux["uncertainty_logits"],
                batch["dino_logits"],
                target,
                valid,
                boundary_weight=float(config.get("boundary", {}).get("loss_weight", 0.2)),
                uncertainty_weight=float(config.get("uncertainty", {}).get("loss_weight", 0.2)),
                boundary_kernel=int(config.get("boundary", {}).get("kernel", 5)),
                boundary_soft=bool(config.get("boundary", {}).get("soft", False)),
                boundary_sigma=float(config.get("boundary", {}).get("sigma", 1.5)),
                pos_weight=config.get("boundary", {}).get("pos_weight", [5.0, 20.0]),
            )
            click_loss = aux["logits"].new_zeros(())
            if bool((positive > 0).any().item() or (negative > 0).any().item()):
                click_loss = click_consistency_loss(
                    aux["logits"], positive, negative,
                    margin=float(loss_cfg.get("click_logit_margin", 2.0)),
                ) * float(loss_cfg.get("click_consistency_weight", 0.05))
            loss = seg_loss + aux_loss + click_loss
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        clip_norm = float(config.get("train", {}).get("clip_grad_norm", 1.0) or 0.0)
        if clip_norm > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        scaler.step(optimizer)
        scaler.update()
        total["loss"] += float(loss.detach().item())
        total["seg_loss"] += float(seg_loss.detach().item())
        total["aux_loss"] += float(aux_loss.detach().item())
        progress.set_postfix(loss=f"{total['loss'] / step:.4f}", clicks=num_clicks)
    count = max(len(loader), 1)
    result = {key: value / count for key, value in total.items()}
    logger.info("epoch=%d train=%s", epoch, result)
    return result


@torch.no_grad()
def validate(
    model: BoundaryAwareInteractiveRefiner,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    clicks: int,
    policy: bool,
) -> dict[str, float]:
    model.eval()
    click_cfg = config.get("clicks", {})
    metric = PaperDice(
        ignore_index=int(config.get("data", {}).get("ignore_index", 255)),
        threshold=config.get("metric", {}).get("threshold", 0.5),
    )
    boundary_total = 0.0
    uncertainty_mae = 0.0
    batches = 0
    for raw_batch in tqdm(loader, desc=f"val {'policy' if policy else 'oracle'} {clicks}", leave=False):
        batch = _move_batch(raw_batch, device)
        target, valid = masks_to_paper_targets(
            batch["mask"], int(config.get("data", {}).get("ignore_index", 255))
        )
        kwargs: dict[str, Any] = {}
        if not policy:
            points = simulate_click_sequence(
                target,
                torch.sigmoid(batch["dino_logits"]) >= float(click_cfg.get("threshold", 0.5)),
                num_clicks=int(clicks),
                strategy=str(click_cfg.get("oracle_strategy", "farthest")),
            )
            kwargs["positive_points"], kwargs["negative_points"] = points
        result = refine_boundary_with_clicks(
            model,
            batch["image"],
            batch["dino_logits"],
            int(clicks),
            threshold=float(click_cfg.get("threshold", 0.5)),
            radius=int(click_cfg.get("radius", 8)),
            mode=str(click_cfg.get("mode", "gaussian")),
            min_separation=int(click_cfg.get("min_separation", 16)),
            stop_priority=(
                float(click_cfg["stop_priority"])
                if click_cfg.get("stop_priority") is not None else None
            ),
            roi_size=(
                int(config.get("refiner", {}).get("roi_size", 0))
                if int(config.get("refiner", {}).get("roi_size", 0)) > 0 else None
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
        batches += 1
    values = metric.compute()
    prefix = "policy" if policy else "oracle"
    denominator = max(batches, 1)
    return {
        **{f"{prefix}_click{clicks}_{key}": value for key, value in values.items()},
        f"{prefix}_click{clicks}_boundary_f1": boundary_total / denominator,
        f"{prefix}_click{clicks}_uncertainty_mae": uncertainty_mae / denominator,
        f"{prefix}_click{clicks}_batches": float(batches),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_score: float,
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_score": float(best_score),
            "config": config,
        },
        path,
    )


def append_metrics(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def check_storage(root_dir: Path, config: dict[str, Any], logger: logging.Logger) -> None:
    usage = shutil.disk_usage(root_dir)
    free_gb = usage.free / (1024**3)
    minimum = float(config.get("storage", {}).get("min_free_gb", 10.0))
    logger.info("storage free=%.2f GiB minimum=%.2f GiB", free_gb, minimum)
    if free_gb < minimum:
        raise RuntimeError(
            f"refusing to continue because free disk space is {free_gb:.2f} GiB < {minimum:.2f} GiB"
        )


def train_fold(config: dict[str, Any], fold: str, root_dir: Path) -> dict[str, Any]:
    fold_dir = root_dir / fold
    (fold_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    logger = setup_logger(fold_dir / "train.log", f"fa_ubir.{fold}.{time.time_ns()}")
    check_storage(root_dir, config, logger)
    device_name = str(config.get("runtime", {}).get("device", "cuda"))
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    epochs = int(config.get("train", {}).get("epochs", 30))
    metrics_path = fold_dir / "metrics.csv"
    existing_rows: list[dict[str, str]] = []
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            existing_rows = list(csv.DictReader(handle))
    last_epoch = max((int(row["epoch"]) for row in existing_rows if row.get("epoch", "").isdigit()), default=0)
    score_key = f"oracle_click{int(config.get('metric', {}).get('select_clicks', 3))}_paper_macro_dice"
    best_score = max((float(row[score_key]) for row in existing_rows if row.get(score_key)), default=-math.inf)
    best_epoch = next(
        (int(row["epoch"]) for row in existing_rows if row.get(score_key) and float(row[score_key]) == best_score),
        0,
    )
    latest_path = fold_dir / "checkpoints" / "latest.pt"
    best_path = fold_dir / "checkpoints" / "best.pt"
    if last_epoch >= epochs and latest_path.is_file() and best_path.is_file():
        logger.info("fold=%s already complete at epoch=%d", fold, last_epoch)
        return {"fold": fold, "best_epoch": best_epoch, "best_score": best_score, "run_dir": str(fold_dir)}

    train_loader = build_loader(config, fold, "train")
    val_loader = build_loader(config, fold, "val")
    model = build_model(config).to(device)
    criterion = build_segmentation_loss(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("train", {}).get("learning_rate", 2e-4)),
        weight_decay=float(config.get("train", {}).get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(config.get("train", {}).get("min_learning_rate", 1e-6)),
    )
    amp_enabled = bool(config.get("runtime", {}).get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch = last_epoch + 1
    if latest_path.is_file():
        try:
            checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = max(start_epoch, int(checkpoint.get("epoch", 0)) + 1)
            best_score = max(best_score, float(checkpoint.get("best_score", -math.inf)))
            logger.info("resumed fold=%s from epoch=%d", fold, start_epoch - 1)
        except (OSError, RuntimeError, KeyError, ValueError, TypeError) as exc:
            logger.warning("resume skipped for fold=%s: %s", fold, exc)

    writer = SummaryWriter(str(fold_dir / "tensorboard"))
    try:
        # Full click curves are expensive because each click adds another
        # high-resolution refiner pass.  During training use a small set (by
        # default only oracle click3) to select checkpoints; the standalone
        # evaluator computes the complete oracle/policy curve after training.
        eval_clicks = [
            int(item)
            for item in config.get("evaluation", {}).get(
                "train_eval_clicks",
                config.get("clicks", {}).get("eval_clicks", [0, 1, 3, 5]),
            )
        ]
        for epoch in range(start_epoch, epochs + 1):
            check_storage(root_dir, config, logger)
            train_values = train_epoch(model, train_loader, criterion, optimizer, scaler, device, config, epoch, logger)
            row: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": train_values["loss"],
                "train_seg_loss": train_values["seg_loss"],
                "train_aux_loss": train_values["aux_loss"],
                "lr": optimizer.param_groups[0]["lr"],
            }
            for clicks in eval_clicks:
                row.update(validate(model, val_loader, device, config, clicks, policy=False))
                if bool(config.get("evaluation", {}).get("run_policy_during_training", False)):
                    row.update(validate(model, val_loader, device, config, clicks, policy=True))
            scheduler.step()
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(key, value, epoch)
            append_metrics(metrics_path, row)
            score = float(row.get(score_key, -math.inf))
            if score > best_score:
                best_score = score
                best_epoch = epoch
                save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_score, config)
            save_checkpoint(latest_path, model, optimizer, scheduler, epoch, best_score, config)
            logger.info("epoch=%d best_epoch=%d best_score=%.6f", epoch, best_epoch, best_score)
    finally:
        writer.close()
        del train_loader, val_loader, model, optimizer, scheduler
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"fold": fold, "best_epoch": best_epoch, "best_score": best_score, "run_dir": str(fold_dir)}


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = resolve_config(load_config(args.config), args)
    seed_everything(int(config.get("project", {}).get("seed", 42)))
    root_dir = project_path(config.get("outputs", {}).get("root", "outputs/interactive_boundary_runs")) / str(
        config.get("project", {}).get("name", "fa_ubir_v1")
    )
    root_dir.mkdir(parents=True, exist_ok=True)
    (root_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    rows: list[dict[str, Any]] = []
    for fold in config.get("train", {}).get("folds_to_run", ["f1", "f2", "f3", "f4", "f5"]):
        rows.append(train_fold(config, str(fold), root_dir))
        write_summary(root_dir / "fold_summary.csv", rows)


if __name__ == "__main__":
    main()
