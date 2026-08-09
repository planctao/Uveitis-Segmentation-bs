from __future__ import annotations

import argparse
import csv
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

from bs.click_simulator import build_pseudo_sam_candidate, build_refiner_features, simulate_click_heatmaps  # noqa: E402
from bs.interactive_refiner import InteractiveResidualRefiner  # noqa: E402
from bs.multilabel import AsymmetricFocalTverskyBCE, PaperDice, masks_to_paper_targets  # noqa: E402
from bs.paths import project_path  # noqa: E402


class CachedDinoDataset(Dataset):
    def __init__(self, manifest_path: Path) -> None:
        self.rows: list[dict[str, str]] = []
        with manifest_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append(row)
        if not self.rows:
            raise RuntimeError(f"No cached samples found in {manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        item = torch.load(row["path"], map_location="cpu", weights_only=True)
        return {
            "sample_id": item["sample_id"],
            "fold": item["fold"],
            "image": item["image"].float(),
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
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with project_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = {key: dict(value) if isinstance(value, dict) else value for key, value in config.items()}
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["runtime"]["num_workers"] = args.num_workers
    if args.learning_rate is not None:
        config["train"]["learning_rate"] = args.learning_rate
    if args.device is not None:
        config["runtime"]["device"] = args.device
    if args.fold is not None:
        config["train"]["folds_to_run"] = [args.fold]
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
    dataset = CachedDinoDataset(manifest)
    return DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=split == "train",
        num_workers=int(config["runtime"].get("num_workers", 4)),
        pin_memory=True,
        drop_last=split == "train",
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
    return InteractiveResidualRefiner(
        in_channels=int(model_cfg.get("in_channels", 13)),
        out_channels=int(model_cfg.get("out_channels", 2)),
        base_channels=int(model_cfg.get("base_channels", 32)),
        residual_scale=float(model_cfg.get("residual_scale", 1.0)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )


def make_inputs(batch: dict[str, Tensor], config: dict[str, Any], num_clicks: int) -> tuple[Tensor, Tensor, Tensor]:
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
        radius=int(click_cfg.get("radius", 8)),
        mode=str(click_cfg.get("mode", "disk")),
    )
    candidate = build_pseudo_sam_candidate(probs, positive, negative, threshold=float(click_cfg.get("threshold", 0.5)))
    features = build_refiner_features(image, dino_logits, candidate, positive, negative)
    return features, dino_logits, mask


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
    clicks = features[:, -4:].amax(dim=1, keepdim=True).clamp(0.0, 1.0)
    if not bool((clicks > 0).any()):
        return final_logits.new_zeros(())
    valid = valid.expand_as(target).float()
    pixel_weight = (1.0 + float(config["loss"].get("click_area_gain", 4.0)) * clicks).expand_as(target) * valid
    bce = F.binary_cross_entropy_with_logits(final_logits, target.to(final_logits.device), reduction="none")
    return (bce * pixel_weight.to(final_logits.device)).sum() / pixel_weight.to(final_logits.device).sum().clamp_min(1.0) * weight


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
        features, dino_logits, mask = make_inputs(batch, config, num_clicks)
        with torch.autocast(device_type=device.type, enabled=bool(config["runtime"].get("amp", True)) and device.type == "cuda"):
            final_logits = model(features, dino_logits)
            loss = criterion(final_logits, mask)
            loss = loss + consistency_loss(final_logits, dino_logits, config)
            loss = loss + click_bce_loss(final_logits, mask, features, config)
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
        features, dino_logits, mask = make_inputs(batch, config, num_clicks)
        final_logits = model(features, dino_logits)
        loss = criterion(final_logits, mask)
        total += float(loss.detach().item())
        metrics.update(final_logits.detach().cpu(), mask.detach().cpu())
    result = {f"click{num_clicks}_loss": total / max(len(loader), 1)}
    for key, value in metrics.compute().items():
        result[f"click{num_clicks}_{key}"] = value
    return result


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_score: float, config: dict[str, Any]) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_score": best_score,
            "config": config,
        },
        path,
    )


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
    writer = SummaryWriter(str(fold_dir / "tensorboard"))
    device = torch.device(config["runtime"].get("device", "cuda") if torch.cuda.is_available() else "cpu")
    train_loader = build_loader(config, fold, "train")
    val_loader = build_loader(config, fold, "val")
    model = build_model(config).to(device)
    criterion = build_loss(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["train"]["learning_rate"]), weight_decay=float(config["train"].get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(config["train"]["epochs"])),
        eta_min=float(config["train"].get("min_learning_rate", 1e-6)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["runtime"].get("amp", True)) and device.type == "cuda")
    best_score = -math.inf
    best_epoch = 0
    eval_clicks = [int(v) for v in config["clicks"].get("eval_clicks", [0, 1, 3, 5])]
    logger.info("fold=%s train_samples=%d val_samples=%d", fold, len(train_loader.dataset), len(val_loader.dataset))
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train = train_epoch(model, train_loader, criterion, optimizer, scaler, device, config, epoch, logger)
        row: dict[str, Any] = {"epoch": epoch, "train_loss": train["loss"], "lr": optimizer.param_groups[0]["lr"]}
        for clicks in eval_clicks:
            row.update(validate_clicks(model, val_loader, criterion, device, config, clicks))
        scheduler.step()
        for key, value in row.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(key, value, epoch)
        append_metrics(fold_dir / "metrics.csv", row)
        score_key = f"click{int(config['metric'].get('select_clicks', 3))}_paper_macro_dice"
        score = float(row.get(score_key, row.get("click3_paper_macro_dice", -math.inf)))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            save_checkpoint(fold_dir / "checkpoints" / "best.pt", model, optimizer, epoch, best_score, config)
        save_checkpoint(fold_dir / "checkpoints" / "latest.pt", model, optimizer, epoch, best_score, config)
        logger.info("epoch=%d best_epoch=%d best_score=%.6f row=%s", epoch, best_epoch, best_score, row)
    writer.close()
    return {"fold": fold, "best_epoch": best_epoch, "best_score": best_score, "run_dir": str(fold_dir)}


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
