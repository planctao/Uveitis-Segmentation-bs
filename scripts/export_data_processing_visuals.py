"""导出专利用数据处理与增强可视化图。

示例：
    PYTHONPATH=src python scripts/export_data_processing_visuals.py \
        --config configs/dinov3_convnext_tiny_multilabel_itksnap.yaml \
        --fold f1 --output-dir runs/patent_data_processing_vis
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bs.augmentations import build_augmentations, denormalize
from bs.dataset import UveitisSegmentationDataset, decode_mask_array, discover_samples
from bs.paths import project_path

PALETTE = {
    0: np.array([0, 0, 0], dtype=np.uint8),
    1: np.array([255, 64, 64], dtype=np.uint8),
    2: np.array([64, 210, 110], dtype=np.uint8),
    3: np.array([80, 150, 255], dtype=np.uint8),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export data processing and augmentation visualizations.")
    parser.add_argument("--config", default="configs/dinov3_convnext_tiny_multilabel_itksnap.yaml")
    parser.add_argument("--fold", default="f1", choices=["f1", "f2", "f3", "f4", "f5"])
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--output-dir", default="runs/patent_data_processing_vis")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_mask(path: Path) -> np.ndarray:
    if path.name.lower().endswith((".nii.gz", ".nii")):
        array = np.asanyarray(nib.load(str(path)).dataobj)
    else:
        array = np.asarray(Image.open(path))
    return decode_mask_array(array, path)


def read_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def colorize_itksnap_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label, color in PALETTE.items():
        out[mask == label] = color
    return out


def colorize_two_region_mask(mask: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = mask
    lesion_1 = (mask_np == 1) | (mask_np == 3)
    lesion_2 = (mask_np == 2) | (mask_np == 3)
    out = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    out[lesion_1] = np.array([255, 64, 64], dtype=np.uint8)
    out[lesion_2] = np.array([64, 210, 110], dtype=np.uint8)
    out[lesion_1 & lesion_2] = np.array([80, 150, 255], dtype=np.uint8)
    return out


def two_binary_masks(mask: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    mask_np = mask.detach().cpu().numpy()
    return ((mask_np == 1) | (mask_np == 3)).astype(np.float32), ((mask_np == 2) | (mask_np == 3)).astype(np.float32)


def overlay_mask(image: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    mask = mask_rgb.astype(np.float32) / 255.0
    foreground = mask.sum(axis=2, keepdims=True) > 0
    return np.where(foreground, image * (1.0 - alpha) + mask * alpha, image).clip(0, 1)


def tensor_to_image(image: torch.Tensor) -> np.ndarray:
    return denormalize(image.detach().cpu()).numpy().transpose(1, 2, 0).clip(0, 1)


def choose_sample(samples: list[Any], sample_id: str | None) -> Any:
    if sample_id:
        for sample in samples:
            if sample.sample_id == sample_id:
                return sample
        raise ValueError(f"sample_id not found: {sample_id}")

    candidates = []
    for sample in samples:
        mask = read_mask(sample.mask_path)
        lesion_1 = (mask == 1) | (mask == 3)
        lesion_2 = (mask == 2) | (mask == 3)
        l1_pixels = int(lesion_1.sum())
        l2_pixels = int(lesion_2.sum())
        overlap_pixels = int((mask == 3).sum())
        if l1_pixels > 0 and l2_pixels > 0:
            candidates.append((l2_pixels, min(l1_pixels, l2_pixels), overlap_pixels, l1_pixels, sample))
    if not candidates:
        raise RuntimeError("No sample with both lesion_1 and lesion_2 found.")
    candidates.sort(reverse=True, key=lambda item: (item[0], item[1], item[2], item[3]))
    return candidates[0][4]


def save_panel(path: Path, panels: list[tuple[str, np.ndarray, str | None]], cols: int = 4) -> None:
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.4 * rows), dpi=180)
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (title, image, cmap) in zip(axes.flat, panels):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def augmentation_cfg(name: str, **kwargs: Any) -> dict[str, Any]:
    cfg = {"enabled": True, "pipeline": [{"name": name, "prob": 1.0}]}
    cfg["pipeline"][0].update(kwargs)
    return cfg


def apply_aug(image: torch.Tensor, mask: torch.Tensor, cfg: dict[str, Any], seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    aug = build_augmentations(cfg)
    if aug is None:
        return image, mask
    return aug(image.clone(), mask.clone())


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with project_path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    data_cfg = config["data"]
    samples = discover_samples(
        dataset_root=project_path(data_cfg["root"]),
        folds=[args.fold],
        image_dir=data_cfg["image_dir"],
        mask_dir=data_cfg["mask_dir"],
        hrnet_result_dir=data_cfg.get("hrnet_result_dir", "HRNet_Result"),
        image_extensions=data_cfg["image_extensions"],
        mask_extensions=data_cfg["mask_extensions"],
        result_extensions=data_cfg.get("result_extensions", data_cfg["image_extensions"]),
        exclude_augmented=True,
    )
    sample = choose_sample(samples, args.sample_id)
    dataset = UveitisSegmentationDataset(
        samples=[sample],
        image_size=tuple(config["train"]["image_size"]),
        label_values=data_cfg["label_values"],
        ignore_index=data_cfg["ignore_index"],
        augment=False,
        augmentation_config=None,
        preprocess_config=config.get("preprocess"),
    )
    item = dataset[0]
    resized_image = tensor_to_image(item["image"])
    resized_mask = item["mask"]
    raw_image = read_image(sample.image_path)
    raw_mask = read_mask(sample.mask_path)
    raw_itk_mask = colorize_itksnap_mask(raw_mask)
    two_region = colorize_two_region_mask(resized_mask)
    overlay = overlay_mask(resized_image, two_region)
    l1_mask, l2_mask = two_binary_masks(resized_mask)

    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray((raw_image * 255).astype(np.uint8)).save(output_dir / f"{sample.sample_id}_01_raw_image.png")
    Image.fromarray(raw_itk_mask).save(output_dir / f"{sample.sample_id}_02_itksnap_palette_mask.png")
    Image.fromarray((resized_image * 255).astype(np.uint8)).save(output_dir / f"{sample.sample_id}_03_resized_normalized_display.png")
    Image.fromarray(two_region).save(output_dir / f"{sample.sample_id}_04_two_region_mask_red_green.png")
    Image.fromarray((overlay * 255).astype(np.uint8)).save(output_dir / f"{sample.sample_id}_05_overlay_red_green.png")

    processing_panels = [
        ("1 Raw FA image", raw_image, None),
        ("2 ITK-SNAP mask", raw_itk_mask, None),
        ("3 Resize + normalize", resized_image, None),
        ("4 Lesion-1 mask", l1_mask, "gray"),
        ("5 Lesion-2 mask", l2_mask, "gray"),
        ("6 Two-region mask", two_region, None),
        ("7 Mask overlay", overlay, None),
    ]
    save_panel(output_dir / f"{sample.sample_id}_data_processing_overview.png", processing_panels, cols=4)

    aug_configs = [
        ("Horizontal flip", augmentation_cfg("hflip")),
        ("Vertical flip", augmentation_cfg("vflip")),
        ("Affine", augmentation_cfg("affine", strength=1.0, degrees=[-25, 25], scale=[0.9, 1.1], translate=0.04)),
        (
            "Foreground crop",
            augmentation_cfg(
                "foreground_resized_crop",
                strength=1.0,
                scale=[0.88, 1.0],
                foreground_labels=[1, 2, 3],
                ignore_index=255,
                min_keep=0.7,
                attempts=8,
            ),
        ),
        ("Brightness/contrast", augmentation_cfg("brightness_contrast", strength=0.75, brightness=[-0.1, 0.1], contrast=[0.85, 1.2])),
        ("Gamma", augmentation_cfg("gamma", strength=0.75, gamma=[0.8, 1.25])),
        ("Gaussian noise", augmentation_cfg("gaussian_noise", strength=0.75, std=[0.0, 0.025])),
        ("Gaussian blur", augmentation_cfg("gaussian_blur", strength=0.75, kernel_size=5, sigma=[0.2, 0.8])),
    ]
    aug_panels = [("Original", resized_image, None)]
    aug_overlay_panels = [("Original overlay", overlay, None)]
    for idx, (name, cfg) in enumerate(aug_configs, start=1):
        aug_image, aug_mask = apply_aug(item["image"], resized_mask, cfg, args.seed + idx)
        aug_img_np = tensor_to_image(aug_image)
        aug_mask_rgb = colorize_two_region_mask(aug_mask)
        aug_panels.append((name, aug_img_np, None))
        aug_overlay_panels.append((name, overlay_mask(aug_img_np, aug_mask_rgb), None))
        Image.fromarray((aug_img_np * 255).astype(np.uint8)).save(output_dir / f"{sample.sample_id}_aug_{idx:02d}_{name.lower().replace('/', '_').replace(' ', '_')}.png")

    for i in range(1, 5):
        aug_image, aug_mask = apply_aug(item["image"], resized_mask, config["augmentations"], args.seed + 100 + i)
        aug_img_np = tensor_to_image(aug_image)
        aug_mask_rgb = colorize_two_region_mask(aug_mask)
        aug_panels.append((f"Random pipeline {i}", aug_img_np, None))
        aug_overlay_panels.append((f"Random pipeline {i}", overlay_mask(aug_img_np, aug_mask_rgb), None))

    save_panel(output_dir / f"{sample.sample_id}_augmentation_images_overview.png", aug_panels, cols=4)
    save_panel(output_dir / f"{sample.sample_id}_augmentation_overlay_overview.png", aug_overlay_panels, cols=4)

    print(f"sample_id={sample.sample_id}")
    print(f"image={sample.image_path}")
    print(f"mask={sample.mask_path}")
    print(f"raw_shape={raw_image.shape[:2]} resized_shape={resized_image.shape[:2]}")
    print(f"labels={sorted(int(x) for x in np.unique(raw_mask))}")
    print(f"lesion1_pixels={int(l1_mask.sum())} lesion2_pixels={int(l2_mask.sum())}")
    print(f"done -> {output_dir}")


if __name__ == "__main__":
    main()
