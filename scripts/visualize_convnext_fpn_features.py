"""可视化 ConvNeXt-Tiny 输入 FPN 的四级 backbone 特征图。

示例：
    PYTHONPATH=src python scripts/visualize_convnext_fpn_features.py \
        --config configs/dinov3_convnext_tiny_multilabel_itksnap.yaml \
        --checkpoint runs/diffleak_f1_baseline_clean/f1/checkpoints/best.pt \
        --fold f1 --num-samples 4 \
        --output-dir runs/convnext_fpn_feature_vis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bs.convnext_seg import DinoV3ConvNeXtSegmentationModel
from bs.dataset import UveitisSegmentationDataset, discover_samples
from bs.paths import project_path

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize four ConvNeXt stage features before FPN fusion.")
    parser.add_argument("--config", default="configs/dinov3_convnext_tiny_multilabel_itksnap.yaml")
    parser.add_argument("--checkpoint", default="runs/diffleak_f1_baseline_clean/f1/checkpoints/best.pt")
    parser.add_argument("--fold", default="f1", choices=["f1", "f2", "f3", "f4", "f5"])
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--sample-id", default=None, help="指定单个 sample_id；默认优先选择含 lesion_2 的样本。")
    parser.add_argument("--output-dir", default="runs/convnext_fpn_feature_vis")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cmap", default="magma")
    parser.add_argument("--overlay-alpha", type=float, default=0.55)
    return parser.parse_args()


def build_model(config: dict) -> DinoV3ConvNeXtSegmentationModel:
    model_cfg = config["model"]
    return DinoV3ConvNeXtSegmentationModel(
        dinov3_code_dir=project_path(model_cfg["dinov3_code_dir"]),
        weights_path=project_path(model_cfg["backbone_weights"]),
        variant=str(model_cfg.get("variant", "tiny")),
        decoder_channels=int(model_cfg.get("decoder_channels", 192)),
        freeze_backbone=False,
        decoder_attention=str(model_cfg.get("decoder_attention", "none")),
        decoder_attention_reduction=int(model_cfg.get("decoder_attention_reduction", 16)),
        decoder_deep_supervision=bool(model_cfg.get("decoder_deep_supervision", False)),
        head_type="conv",
    )


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().numpy().transpose(1, 2, 0)
    array = array * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(array, 0.0, 1.0)


def normalize_map(array: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(array, [1, 99])
    array = np.clip(array, lo, hi)
    return (array - array.min()) / (np.ptp(array) + 1e-8)


def feature_activation(feature: torch.Tensor, output_size: tuple[int, int]) -> np.ndarray:
    activation = feature.detach().float().pow(2).mean(dim=1, keepdim=True).sqrt()
    activation = F.interpolate(activation, size=output_size, mode="bilinear", align_corners=False)
    return normalize_map(activation[0, 0].cpu().numpy())


def make_gt_overlay(mask: torch.Tensor) -> np.ndarray:
    lesion_1 = ((mask == 1) | (mask == 3)).detach().cpu().numpy().astype(np.float32)
    lesion_2 = ((mask == 2) | (mask == 3)).detach().cpu().numpy().astype(np.float32)
    overlay = np.zeros((*lesion_1.shape, 3), dtype=np.float32)
    overlay[..., 0] = lesion_1
    overlay[..., 2] = lesion_2
    overlay[..., 1] = 0.4 * (lesion_1 * lesion_2)
    return overlay


def choose_indices(dataset: UveitisSegmentationDataset, sample_id: str | None, count: int) -> list[int]:
    if sample_id is not None:
        for idx, sample in enumerate(dataset.samples):
            if sample.sample_id == sample_id:
                return [idx]
        raise ValueError(f"sample_id not found in fold: {sample_id}")

    lesion2, lesion1, background = [], [], []
    for idx in range(len(dataset)):
        mask = dataset[idx]["mask"]
        has_l1 = bool(torch.any((mask == 1) | (mask == 3)))
        has_l2 = bool(torch.any((mask == 2) | (mask == 3)))
        if has_l2:
            lesion2.append(idx)
        elif has_l1:
            lesion1.append(idx)
        else:
            background.append(idx)
        if len(lesion2) >= count:
            break
    chosen = (lesion2 + lesion1 + background)[:count]
    if not chosen:
        raise RuntimeError("No samples available for visualization.")
    return chosen


def save_individual_heatmap(path: Path, image: np.ndarray, heatmap: np.ndarray, cmap: str, alpha: float, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4, 4), dpi=180)
    ax.imshow(image)
    ax.imshow(heatmap, cmap=cmap, alpha=alpha)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout(pad=0.05)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    args = parse_args()
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
    dataset = UveitisSegmentationDataset(
        samples=samples,
        image_size=tuple(config["train"]["image_size"]),
        label_values=data_cfg["label_values"],
        ignore_index=data_cfg["ignore_index"],
        augment=False,
        augmentation_config=None,
        preprocess_config=config.get("preprocess"),
    )

    device = torch.device(args.device)
    model = build_model(config).to(device)
    checkpoint = torch.load(project_path(args.checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint, strict=True)
    model.eval()

    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = choose_indices(dataset, args.sample_id, int(args.num_samples))
    print(f"selected sample_ids: {[dataset.samples[i].sample_id for i in indices]}")

    for index in indices:
        item = dataset[index]
        image_tensor = item["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.extract_multiscale_features(image_tensor)

        image_np = denormalize_image(item["image"])
        output_size = tuple(item["image"].shape[-2:])
        heatmaps = [feature_activation(feature, output_size) for feature in features]
        stage_titles = [f"Stage {i + 1}  {tuple(feature.shape[-2:])}" for i, feature in enumerate(features)]

        panels = [("FA image", image_np, None), ("GT: L1 red / L2 blue", make_gt_overlay(item["mask"]), None)]
        panels.extend((title, heatmap, args.cmap) for title, heatmap in zip(stage_titles, heatmaps))

        fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.4), dpi=180)
        for ax, (title, data, cmap) in zip(axes, panels):
            if cmap is None:
                ax.imshow(data)
            else:
                ax.imshow(image_np)
                ax.imshow(data, cmap=cmap, alpha=args.overlay_alpha)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.suptitle(f"ConvNeXt-Tiny four stage features before FPN: {item['sample_id']}", fontsize=11)
        fig.tight_layout()
        panel_path = output_dir / f"{item['sample_id']}_fpn_stage_features.png"
        fig.savefig(panel_path, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

        for i, heatmap in enumerate(heatmaps, start=1):
            save_individual_heatmap(
                output_dir / f"{item['sample_id']}_stage{i}_fpn_input.png",
                image_np,
                heatmap,
                args.cmap,
                args.overlay_alpha,
                f"Stage {i} FPN input activation",
            )
        print(f"saved {panel_path}")

    print(f"done -> {output_dir}")


if __name__ == "__main__":
    main()
