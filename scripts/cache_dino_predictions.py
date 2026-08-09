from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from bs.dataset import UveitisSegmentationDataset, discover_samples  # noqa: E402
from bs.paths import project_path  # noqa: E402
from train_dinov3_multilabel import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen DINOv3 predictions for interactive refiner training.")
    parser.add_argument("--config", default="configs/dinov3_convnext_tiny_vsubr_vw05.yaml")
    parser.add_argument("--checkpoint-template", default="runs/vsubr_vw05_{fold}/{fold}/checkpoints/best.pt")
    parser.add_argument("--output-root", default="outputs/dino_refiner_cache/vsubr_vw05")
    parser.add_argument("--folds", default="f1,f2,f3,f4,f5")
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_yaml_config(path: str) -> dict[str, Any]:
    with project_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def load_checkpoint_config(path: Path, fallback_config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("config") or fallback_config
    state_dict = checkpoint.get("model", checkpoint)
    return config, state_dict


def build_cache_loader(config: dict[str, Any], val_fold: str, split: str, batch_size: int, num_workers: int, max_samples: int | None) -> DataLoader:
    data_cfg = config["data"]
    train_cfg = config["train"]
    folds = [val_fold] if split == "val" else [fold for fold in data_cfg["folds"] if fold != val_fold]
    samples = discover_samples(
        dataset_root=project_path(data_cfg["root"]),
        folds=folds,
        image_dir=data_cfg["image_dir"],
        mask_dir=data_cfg["mask_dir"],
        hrnet_result_dir=data_cfg.get("hrnet_result_dir", "HRNet_Result"),
        image_extensions=data_cfg["image_extensions"],
        mask_extensions=data_cfg["mask_extensions"],
        result_extensions=data_cfg.get("result_extensions", data_cfg["image_extensions"]),
        exclude_augmented=bool(data_cfg.get("exclude_val_augmented", True)),
    )
    if max_samples is not None:
        samples = samples[: int(max_samples)]
    dataset = UveitisSegmentationDataset(
        samples=samples,
        image_size=tuple(train_cfg["image_size"]),
        label_values=data_cfg["label_values"],
        ignore_index=data_cfg["ignore_index"],
        augment=False,
        preprocess_config=config.get("preprocess"),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def save_batch(batch: dict[str, Any], logits: torch.Tensor, output_dir: Path, rows: list[dict[str, str]]) -> None:
    images = batch["image"].detach().cpu().to(torch.float16)
    masks = batch["mask"].detach().cpu().to(torch.uint8)
    logits = logits.detach().cpu().to(torch.float16)
    sample_ids = list(batch["sample_id"])
    folds = list(batch["fold"])
    for idx, sample_id in enumerate(sample_ids):
        filename = f"{safe_name(str(sample_id))}.pt"
        path = output_dir / filename
        torch.save(
            {
                "sample_id": str(sample_id),
                "fold": str(folds[idx]),
                "image": images[idx],
                "mask": masks[idx],
                "dino_logits": logits[idx],
            },
            path,
        )
        rows.append({"sample_id": str(sample_id), "fold": str(folds[idx]), "path": str(path)})


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fallback_config = load_yaml_config(args.config)
    folds = [item.strip() for item in args.folds.split(",") if item.strip()]
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    output_root = project_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for fold in folds:
        ckpt_path = project_path(args.checkpoint_template.format(fold=fold))
        config, state_dict = load_checkpoint_config(ckpt_path, fallback_config)
        model = build_model(config).to(device)
        result = model.load_state_dict(state_dict, strict=True)
        logging.info("loaded %s missing=%d unexpected=%d", ckpt_path, len(result.missing_keys), len(result.unexpected_keys))
        model.eval()
        for split in splits:
            max_samples = args.max_train_samples if split == "train" else args.max_val_samples
            loader = build_cache_loader(config, fold, split, args.batch_size, args.num_workers, max_samples)
            split_dir = output_root / fold / split
            split_dir.mkdir(parents=True, exist_ok=True)
            rows: list[dict[str, str]] = []
            with torch.no_grad():
                for batch in tqdm(loader, desc=f"cache {fold} {split}"):
                    images = batch["image"].to(device, non_blocking=True)
                    logits = model(images)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    save_batch(batch, logits, split_dir, rows)
            manifest = output_root / fold / f"{split}_manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["sample_id", "fold", "path"])
                writer.writeheader()
                writer.writerows(rows)
            logging.info("wrote %d cached samples to %s", len(rows), split_dir)


if __name__ == "__main__":
    main()
