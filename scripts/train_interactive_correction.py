"""Run the FA-ClickPCM B0--B4 small ablation on one fold.

The experiment is deliberately compact: B0 is the frozen DINO coarse mask;
B1--B4 train the same small mask-guided refiner with progressively more
local/robust updates.  Existing DINO cache files are read in place and no
prediction cache is generated.
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
    parser.add_argument("--config", default="configs/fa_clickpcm_ablation.yaml")
    parser.add_argument("--fold", default=None)
    parser.add_argument("--variant", default="all", choices=["all", "b0", "b1", "b2", "b3", "b4"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with project_path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("configuration must be a mapping")
    return value


def resolve_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(json.dumps(config))
    if args.epochs is not None:
        config["train"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        config["train"]["batch_size"] = int(args.batch_size)
    if args.max_train_samples is not None:
        config["train"]["max_train_samples"] = int(args.max_train_samples)
    if args.max_val_samples is not None:
        config["train"]["max_val_samples"] = int(args.max_val_samples)
    if args.device is not None:
        config["runtime"]["device"] = args.device
    if args.seed is not None:
        config["project"]["seed"] = int(args.seed)
    if args.fold is not None:
        config["train"]["folds_to_run"] = [args.fold]
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"clickpcm.{time.time_ns()}")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    return logger


def check_storage(path: Path, minimum_gb: float, logger: logging.Logger) -> None:
    usage = shutil.disk_usage(path)
    free = usage.free / (1024**3)
    logger.info("disk_free_gib=%.2f minimum_gib=%.2f", free, minimum_gb)
    if free < float(minimum_gb):
        raise RuntimeError(f"free disk space {free:.2f} GiB below safety limit {minimum_gb:.2f} GiB")


def build_loader(config: dict[str, Any], fold: str, split: str) -> DataLoader:
    manifest = project_path(config["cache"]["root"]) / fold / f"{split}_manifest.csv"
    limit = config["train"].get("max_train_samples" if split == "train" else "max_val_samples")
    dataset = CachedDinoDataset(manifest, limit)
    return DataLoader(
        dataset,
        batch_size=int(config["train"].get("batch_size", 2)),
        shuffle=split == "train",
        num_workers=int(config["runtime"].get("num_workers", 0)),
        pin_memory=True,
        persistent_workers=False,
    )


def batch_click_maps(
    target: Tensor,
    current_logits: Tensor,
    num_clicks: int,
    *,
    threshold: float,
    radius: int,
    mode: str,
    strategy: str,
    jitter: float,
    dropout: float,
) -> tuple[Tensor, Tensor]:
    points = simulate_click_points(
        target, torch.sigmoid(current_logits) >= float(threshold), num_clicks=num_clicks, strategy=strategy
    )
    height, width = current_logits.shape[-2:]
    pos = perturb_click_points(points[0], height, width, jitter=jitter, dropout=dropout)
    neg = perturb_click_points(points[1], height, width, jitter=jitter, dropout=dropout)
    return (
        click_points_to_heatmaps(pos, height, width, radius=radius, mode=mode).to(current_logits.dtype),
        click_points_to_heatmaps(neg, height, width, radius=radius, mode=mode).to(current_logits.dtype),
    )


def build_loss(config: dict[str, Any]) -> AsymmetricFocalTverskyBCE:
    loss = config["loss"]
    return AsymmetricFocalTverskyBCE(
        pos_weight=loss.get("pos_weight", [3.0, 60.0]),
        bce_weight=float(loss.get("bce_weight", 0.5)),
        tversky_weight=float(loss.get("tversky_weight", 1.0)),
        alpha=float(loss.get("tversky_alpha", 0.2)),
        beta=float(loss.get("tversky_beta", 0.8)),
        gamma=float(loss.get("focal_gamma", 0.75)),
        ignore_index=int(config["data"].get("ignore_index", 255)),
    )


def run_variant(
    variant: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    root: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    variant_dir = root / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = variant_dir / "metrics.csv"
    if variant == "b0":
        values = evaluate_model(None, val_loader, config, device, variant)
        row = {"epoch": 0, **values}
        write_metrics(metrics_path, [row])
        return {"variant": variant, "best_epoch": 0, **values, "run_dir": str(variant_dir)}

    model = ClickPCMRefiner(
        base_channels=int(config["model"].get("base_channels", 8)),
        dropout=float(config["model"].get("dropout", 0.1)),
    ).to(device)
    criterion = build_loss(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"].get("learning_rate", 2e-4)),
        weight_decay=float(config["train"].get("weight_decay", 1e-4)),
    )
    epochs = int(config["train"].get("epochs", 5))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-6)
    amp = bool(config["runtime"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    click_cfg = config["clicks"]
    train_choices = [int(v) for v in click_cfg.get("train_clicks", [1, 3])]
    best = -1.0
    best_epoch = 0
    rows: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        check_storage(root, float(config["storage"].get("min_free_gb", 10.0)), logger)
        model.train()
        total = 0.0
        progress = tqdm(train_loader, desc=f"{variant} train {epoch}", leave=False)
        for step, raw in enumerate(progress, start=1):
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in raw.items()}
            target, _ = masks_to_paper_targets(batch["mask"], int(config["data"].get("ignore_index", 255)))
            clicks = random.choice(train_choices)
            jitter = float(click_cfg.get("jitter", 0.0)) if variant == "b4" else 0.0
            dropout = float(click_cfg.get("dropout", 0.0)) if variant == "b4" else 0.0
            mode = "gaussian" if variant == "b4" else "disk"
            positive, negative = batch_click_maps(
                target, batch["dino_logits"], clicks,
                threshold=float(click_cfg.get("threshold", 0.5)),
                radius=int(click_cfg.get("radius", 8)), mode=mode,
                strategy=str(click_cfg.get("strategy", "farthest")), jitter=jitter, dropout=dropout,
            )
            with torch.autocast(device_type=device.type, enabled=amp):
                result = correction_step(
                    model, batch["image"], batch["dino_logits"], positive, negative,
                    variant=variant, threshold=float(click_cfg.get("threshold", 0.5)),
                    crop_size=int(config["model"].get("crop_size", 256)),
                    max_delta=float(config["model"].get("max_delta", 2.0)),
                )
                loss = criterion(result.logits, batch["mask"])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            clip_norm = float(config["train"].get("clip_grad_norm", 1.0) or 0.0)
            if clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach().item())
            progress.set_postfix(loss=f"{total / step:.4f}", clicks=clicks)
        scheduler.step()
        values = evaluate_model(model, val_loader, config, device, variant)
        row = {"epoch": epoch, "train_loss": total / max(len(train_loader), 1), "lr": optimizer.param_groups[0]["lr"], **values}
        rows.append(row)
        write_metrics(metrics_path, rows)
        score = float(values.get("click3_paper_macro_dice", values.get("click0_paper_macro_dice", -1.0)))
        if score > best:
            best = score
            best_epoch = epoch
            torch.save({"epoch": epoch, "model": model.state_dict(), "config": config, "best_score": best}, variant_dir / "best.pt")
        torch.save({"epoch": epoch, "model": model.state_dict(), "config": config, "best_score": best}, variant_dir / "latest.pt")
        logger.info("variant=%s epoch=%d score=%.6f metrics=%s", variant, epoch, score, values)
    checkpoint = variant_dir / "best.pt"
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
    final = evaluate_model(model, val_loader, config, device, variant)
    del model, optimizer, scheduler, scaler, criterion
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"variant": variant, "best_epoch": best_epoch, **final, "run_dir": str(variant_dir)}


@torch.no_grad()
def evaluate_model(model: ClickPCMRefiner | None, loader: DataLoader, config: dict[str, Any], device: torch.device, variant: str) -> dict[str, float]:
    if model is not None:
        model.eval()
    click_cfg = config["clicks"]
    results: dict[str, float] = {}
    for num_clicks in [0, 1, 3, 5]:
        metric = PaperDice(ignore_index=int(config["data"].get("ignore_index", 255)), threshold=0.5)
        for raw in tqdm(loader, desc=f"{variant} val {num_clicks}", leave=False):
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in raw.items()}
            current = batch["dino_logits"]
            target, _ = masks_to_paper_targets(batch["mask"], int(config["data"].get("ignore_index", 255)))
            for step in range(num_clicks):
                jitter = float(click_cfg.get("jitter", 0.0)) if variant == "b4" else 0.0
                dropout = float(click_cfg.get("dropout", 0.0)) if variant == "b4" else 0.0
                mode = "gaussian" if variant == "b4" else "disk"
                positive, negative = batch_click_maps(
                    target, current, 1,
                    threshold=float(click_cfg.get("threshold", 0.5)),
                    radius=int(click_cfg.get("radius", 8)), mode=mode,
                    strategy=str(click_cfg.get("strategy", "farthest")), jitter=jitter, dropout=dropout,
                )
                if model is not None and variant != "b0":
                    current = correction_step(
                        model, batch["image"], current, positive, negative,
                        variant=variant, threshold=float(click_cfg.get("threshold", 0.5)),
                        crop_size=int(config["model"].get("crop_size", 256)),
                        max_delta=float(config["model"].get("max_delta", 2.0)),
                    ).logits
            metric.update(current.detach().cpu(), batch["mask"].detach().cpu())
        values = metric.compute()
        for key, value in values.items():
            if key.startswith("paper_") and (key in {"paper_dice_1", "paper_dice_2", "paper_macro_dice"}):
                results[f"click{num_clicks}_{key}"] = float(value)
    return results


def write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = resolve_config(load_config(args.config), args)
    seed_everything(int(config["project"].get("seed", 42)))
    output_root = project_path(config["outputs"]["root"]) / str(config["project"]["name"])
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    logger = setup_logger(output_root / "experiment.log")
    device_name = str(config["runtime"].get("device", "cuda"))
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    all_summary: list[dict[str, Any]] = []
    variants = [args.variant] if args.variant != "all" else ["b0", "b1", "b2", "b3", "b4"]
    for fold in config["train"].get("folds_to_run", ["f1"]):
        train_loader = build_loader(config, str(fold), "train")
        val_loader = build_loader(config, str(fold), "val")
        fold_root = output_root / str(fold)
        fold_root.mkdir(parents=True, exist_ok=True)
        logger.info("starting fold=%s train=%d val=%d variants=%s device=%s", fold, len(train_loader.dataset), len(val_loader.dataset), variants, device)
        for variant in variants:
            check_storage(output_root, float(config["storage"].get("min_free_gb", 10.0)), logger)
            summary = run_variant(variant, train_loader, val_loader, config, device, fold_root, logger)
            summary["fold"] = str(fold)
            all_summary.append(summary)
            with (fold_root / "ablation_summary.json").open("w", encoding="utf-8") as handle:
                json.dump(all_summary, handle, ensure_ascii=False, indent=2)
        del train_loader, val_loader
        gc.collect()
    with (output_root / "ablation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(all_summary, handle, ensure_ascii=False, indent=2)
    logger.info("completed summaries=%s", all_summary)


if __name__ == "__main__":
    main()
