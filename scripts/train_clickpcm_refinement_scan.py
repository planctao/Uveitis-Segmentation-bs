"""C1--C4 refinement scan for FA-ClickPCM.

C1 trains a true cumulative rollout, C2 adds a click-local loss, C3 adds
Gaussian signed/distance prompts, and C4 repeats C3 with a 384-pixel focus
crop.  This script intentionally uses the existing compact DINO cache only.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from train_interactive_refiner import CachedDinoDataset  # noqa: E402
from bs.click_simulator import click_points_to_heatmaps, perturb_click_points, simulate_click_points  # noqa: E402
from bs.interactive_correction import ClickPCMRefiner, correction_step  # noqa: E402
from bs.multilabel import AsymmetricFocalTverskyBCE, PaperDice, masks_to_paper_targets  # noqa: E402
from bs.paths import project_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fa_clickpcm_refinement_scan.yaml")
    parser.add_argument("--variant", default="all", choices=["all", "c1", "c2", "c3", "c4"])
    parser.add_argument("--fold", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with project_path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(json.dumps(config))
    if args.epochs is not None:
        config["train"]["epochs"] = int(args.epochs)
    if args.max_train_samples is not None:
        config["train"]["max_train_samples"] = int(args.max_train_samples)
    if args.max_val_samples is not None:
        config["train"]["max_val_samples"] = int(args.max_val_samples)
    if args.device is not None:
        config["runtime"]["device"] = args.device
    if args.fold is not None:
        config["train"]["folds_to_run"] = [args.fold]
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"clickpcm_scan.{time.time_ns()}")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in [logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def check_storage(path: Path, minimum_gb: float, logger: logging.Logger) -> None:
    free = shutil.disk_usage(path).free / (1024**3)
    logger.info("disk_free_gib=%.2f minimum_gib=%.2f", free, minimum_gb)
    if free < float(minimum_gb):
        raise RuntimeError(f"free disk space {free:.2f} GiB below safety limit {minimum_gb:.2f} GiB")


def loader(config: dict[str, Any], fold: str, split: str) -> DataLoader:
    manifest = project_path(config["cache"]["root"]) / fold / f"{split}_manifest.csv"
    limit = config["train"].get("max_train_samples" if split == "train" else "max_val_samples")
    dataset = CachedDinoDataset(manifest, limit)
    return DataLoader(dataset, batch_size=int(config["train"].get("batch_size", 2)),
                      shuffle=split == "train", num_workers=0, pin_memory=True)


def click_maps(target: Tensor, current: Tensor, clicks: int, config: dict[str, Any], variant: str) -> tuple[Tensor, Tensor]:
    cfg = config["clicks"]
    points = simulate_click_points(target, torch.sigmoid(current) >= float(cfg.get("threshold", 0.5)),
                                   num_clicks=clicks, strategy=str(cfg.get("strategy", "farthest")))
    height, width = current.shape[-2:]
    mode = "gaussian" if variant in {"c3", "c4"} else "disk"
    pos = perturb_click_points(points[0], height, width, jitter=0.0, dropout=0.0)
    neg = perturb_click_points(points[1], height, width, jitter=0.0, dropout=0.0)
    return (click_points_to_heatmaps(pos, height, width, radius=int(cfg.get("radius", 8)), mode=mode).to(current.dtype),
            click_points_to_heatmaps(neg, height, width, radius=int(cfg.get("radius", 8)), mode=mode).to(current.dtype))


def build_loss(config: dict[str, Any]) -> AsymmetricFocalTverskyBCE:
    cfg = config["loss"]
    return AsymmetricFocalTverskyBCE(pos_weight=cfg.get("pos_weight", [3.0, 60.0]),
                                     bce_weight=float(cfg.get("bce_weight", 0.5)),
                                     tversky_weight=float(cfg.get("tversky_weight", 1.0)),
                                     alpha=float(cfg.get("tversky_alpha", 0.2)),
                                     beta=float(cfg.get("tversky_beta", 0.8)),
                                     gamma=float(cfg.get("focal_gamma", 0.75)),
                                     ignore_index=int(config["data"].get("ignore_index", 255)))


def local_loss(logits: Tensor, mask: Tensor, positive: Tensor, negative: Tensor, config: dict[str, Any]) -> Tensor:
    target, valid = masks_to_paper_targets(mask, int(config["data"].get("ignore_index", 255)))
    target = target.to(logits.device)
    valid = valid.expand_as(target).to(logits.device).float()
    support = torch.maximum(positive, negative).amax(dim=1, keepdim=True)
    # A 33-pixel halo approximates the local correction region while keeping
    # the loss target-free at deployment.
    halo = F.max_pool2d(support, kernel_size=33, stride=1, padding=16)
    weights = (1.0 + float(config["loss"].get("local_gain", 4.0)) * halo).expand_as(target)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (bce * weights * valid).sum() / (weights * valid).sum().clamp_min(1.0)


def make_model(config: dict[str, Any], variant: str, device: torch.device) -> ClickPCMRefiner:
    add_distance = variant in {"c3", "c4"}
    return ClickPCMRefiner(base_channels=int(config["model"].get("base_channels", 8)),
                           dropout=float(config["model"].get("dropout", 0.1)),
                           in_channels=15 if add_distance else 13).to(device)


def run_rollout(model: ClickPCMRefiner, image: Tensor, dino: Tensor, target: Tensor, mask: Tensor,
                clicks: int, variant: str, config: dict[str, Any], criterion: AsymmetricFocalTverskyBCE | None,
                training: bool) -> tuple[Tensor, Tensor]:
    current = dino
    total = dino.new_zeros(())
    cfg = config["clicks"]
    add_distance = variant in {"c3", "c4"}
    crop_size = int(config["model"].get("crop_size", 256 if variant != "c4" else 384))
    for _ in range(clicks):
        positive, negative = click_maps(target, current, 1, config, variant)
        result = correction_step(model, image, current, positive, negative, variant="b3",
                                 threshold=float(cfg.get("threshold", 0.5)), crop_size=crop_size,
                                 max_delta=float(config["model"].get("max_delta", 2.0)),
                                 add_distance=add_distance)
        current = result.logits
        if training and criterion is not None:
            step_loss = criterion(current, mask)
            if variant in {"c2", "c3", "c4"}:
                step_loss = step_loss + float(config["loss"].get("local_weight", 0.5)) * local_loss(
                    current, mask, positive, negative, config
                )
            total = total + step_loss
        # Truncated BPTT makes the rollout genuinely cumulative without
        # retaining five high-resolution graphs in memory.
        if training:
            current = current.detach()
    return current, total / max(clicks, 1)


@torch.no_grad()
def evaluate(model: ClickPCMRefiner, val_loader: DataLoader, config: dict[str, Any], device: torch.device,
             variant: str) -> dict[str, float]:
    model.eval()
    values: dict[str, float] = {}
    for clicks in [0, 1, 3, 5]:
        metric = PaperDice(ignore_index=int(config["data"].get("ignore_index", 255)), threshold=0.5)
        for raw in tqdm(val_loader, desc=f"{variant} val {clicks}", leave=False):
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in raw.items()}
            target, _ = masks_to_paper_targets(batch["mask"], int(config["data"].get("ignore_index", 255)))
            current, _ = run_rollout(model, batch["image"], batch["dino_logits"], target, batch["mask"], clicks,
                                     variant, config, None, False)
            metric.update(current.cpu(), batch["mask"].cpu())
        result = metric.compute()
        for key in ["paper_dice_1", "paper_dice_2", "paper_macro_dice"]:
            values[f"click{clicks}_{key}"] = float(result[key])
    return values


def run_variant(variant: str, train_loader: DataLoader, val_loader: DataLoader, config: dict[str, Any],
                device: torch.device, out_dir: Path, logger: logging.Logger) -> dict[str, Any]:
    variant_dir = out_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    model = make_model(config, variant, device)
    criterion = build_loss(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["train"].get("learning_rate", 2e-4)),
                                  weight_decay=float(config["train"].get("weight_decay", 1e-4)))
    epochs = int(config["train"].get("epochs", 5))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-6)
    amp = bool(config["runtime"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best, best_epoch = -1.0, 0
    rows: list[dict[str, Any]] = []
    choices = [int(v) for v in config["clicks"].get("train_clicks", [1, 2, 3])]
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for raw in tqdm(train_loader, desc=f"{variant} train {epoch}", leave=False):
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in raw.items()}
            target, _ = masks_to_paper_targets(batch["mask"], int(config["data"].get("ignore_index", 255)))
            clicks = random.choice(choices)
            with torch.autocast(device_type=device.type, enabled=amp):
                _, loss = run_rollout(model, batch["image"], batch["dino_logits"], target, batch["mask"], clicks,
                                       variant, config, criterion, True)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            clip = float(config["train"].get("clip_grad_norm", 1.0) or 0.0)
            if clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach().item())
        scheduler.step()
        metrics = evaluate(model, val_loader, config, device, variant)
        row = {"epoch": epoch, "train_loss": total / max(len(train_loader), 1),
               "lr": optimizer.param_groups[0]["lr"], **metrics}
        rows.append(row)
        write_metrics(variant_dir / "metrics.csv", rows)
        score = float(metrics["click3_paper_macro_dice"])
        if score > best:
            best, best_epoch = score, epoch
            torch.save({"epoch": epoch, "model": model.state_dict(), "config": config, "best_score": best}, variant_dir / "best.pt")
        torch.save({"epoch": epoch, "model": model.state_dict(), "config": config, "best_score": best}, variant_dir / "latest.pt")
        logger.info("variant=%s epoch=%d score=%.6f", variant, epoch, score)
    state = torch.load(variant_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    final = evaluate(model, val_loader, config, device, variant)
    summary = {"variant": variant, "best_epoch": best_epoch, **final, "run_dir": str(variant_dir)}
    del model, criterion, optimizer, scheduler, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = resolve_config(load_config(args.config), args)
    seed_everything(int(config["project"].get("seed", 42)))
    root = project_path(config["outputs"]["root"]) / str(config["project"]["name"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    logger = setup_logger(root / "experiment.log")
    device_name = str(config["runtime"].get("device", "cuda"))
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    variants = [args.variant] if args.variant != "all" else ["c1", "c2", "c3", "c4"]
    summaries: list[dict[str, Any]] = []
    for fold in config["train"].get("folds_to_run", ["f1"]):
        train_loader = loader(config, str(fold), "train")
        val_loader = loader(config, str(fold), "val")
        fold_root = root / str(fold)
        fold_root.mkdir(parents=True, exist_ok=True)
        logger.info("starting fold=%s train=%d val=%d variants=%s device=%s", fold, len(train_loader.dataset), len(val_loader.dataset), variants, device)
        for variant in variants:
            check_storage(root, float(config["storage"].get("min_free_gb", 10.0)), logger)
            summary = run_variant(variant, train_loader, val_loader, config, device, fold_root, logger)
            summary["fold"] = str(fold)
            summaries.append(summary)
            (fold_root / "refinement_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
        del train_loader, val_loader
        gc.collect()
    (root / "refinement_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("completed summaries=%s", summaries)


if __name__ == "__main__":
    main()
