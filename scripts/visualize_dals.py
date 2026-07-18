"""可视化 DALS（扩散外观渗漏合成）效果。

生成一张拼图：
- 前 N 行：真实目标图 -> DALS 合成后 -> 新增 lesion_2 区域(红)
- 末行：机制拆解 -> 采集的真实病灶 crop / 二值足迹 / 热核扩散软 alpha

用法：
    PYTHONPATH=src python scripts/visualize_dals.py
输出：runs/dals_vis/dals_visualization.png
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bs.augmentations import denormalize
from bs.dataset import UveitisSegmentationDataset, discover_samples
from bs.leakage_synthesis import LeakageCopyPaste, build_instance_bank, gaussian_blur
from bs.paths import project_path


def to_disp(image: torch.Tensor) -> np.ndarray:
    return denormalize(image).clamp(0, 1).permute(1, 2, 0).numpy()


def lesion2(mask: torch.Tensor) -> np.ndarray:
    return ((mask == 2) | (mask == 3)).numpy()


def main() -> None:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    root = project_path("dataset/dataset/split_dataorigin")
    samples = discover_samples(root, ["f1"], image_dir="img", mask_dir="mask_only_itksnap")
    bank = build_instance_bank(samples, image_size=(768, 768), lesions=(2,), max_instances=400, min_area=16)
    print(f"instance bank size (lesion_2) = {len(bank)}")
    if not bank:
        raise SystemExit("空实例库：f1 中未找到 lesion_2 实例")

    dataset = UveitisSegmentationDataset(
        samples=samples, image_size=(768, 768), label_values=(0, 1, 2, 3), ignore_index=255, augment=False
    )

    # 优先选“原本没有 lesion_2”的目标图，这样合成新增区域一目了然
    target_indices: list[int] = []
    for index in range(len(dataset)):
        if lesion2(dataset[index]["mask"]).sum() == 0:
            target_indices.append(index)
        if len(target_indices) >= 3:
            break
    if len(target_indices) < 3:
        target_indices = list(range(3))

    block = LeakageCopyPaste(
        prob=1.0,
        instances=bank,
        targets=(2,),
        max_instances=(2, 3),
        scale=(0.8, 1.3),
        diffusion_sigma=(6.0, 12.0),
        intensity_gain=(1.15, 1.4),
        placement="macula_biased",
        alpha_threshold=0.5,
    )

    rows = len(target_indices) + 1
    fig, axes = plt.subplots(rows, 3, figsize=(13, 4.2 * rows))

    for row, index in enumerate(target_indices):
        item = dataset[index]
        image, mask = item["image"], item["mask"]
        before_disp = to_disp(image)
        before_l2 = lesion2(mask)
        aug_image, aug_mask = block.apply(image.clone(), mask.clone())
        after_disp = to_disp(aug_image)
        new_region = lesion2(aug_mask) & (~before_l2)

        overlay = after_disp.copy()
        overlay[new_region] = [1.0, 0.0, 0.0]

        axes[row, 0].imshow(before_disp)
        axes[row, 0].set_title(f"Original ({item['sample_id']})")
        axes[row, 1].imshow(after_disp)
        axes[row, 1].set_title("After DALS synthesis")
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(f"New lesion_2 (red)  +{int(new_region.sum())} px")
        for col in range(3):
            axes[row, col].axis("off")

    # 机制拆解：取一个较大的病灶实例，展示 crop / mask / 热核扩散 alpha
    instance = max(bank, key=lambda inst: float(inst.mask.sum()))
    footprint = instance.mask
    pad = 24
    padded = F.pad(footprint.view(1, 1, *footprint.shape), (pad, pad, pad, pad)).squeeze(0)
    density = gaussian_blur(padded, 10.0).squeeze(0)
    alpha = (density / density.max().clamp_min(1e-6)).clamp(0, 1)
    alpha = torch.maximum(alpha, F.pad(footprint, (pad, pad, pad, pad)))

    last = len(target_indices)
    axes[last, 0].imshow(instance.image.clamp(0, 1).permute(1, 2, 0).numpy())
    axes[last, 0].set_title("Sampled real lesion crop")
    axes[last, 1].imshow(footprint.numpy(), cmap="gray")
    axes[last, 1].set_title("Binary footprint")
    heat = axes[last, 2].imshow(alpha.numpy(), cmap="magma")
    axes[last, 2].set_title("Heat-kernel soft alpha (bright core -> diffuse edge)")
    fig.colorbar(heat, ax=axes[last, 2], fraction=0.046, pad=0.04)
    for col in range(3):
        axes[last, col].axis("off")

    fig.suptitle("DALS: Diffusion-Appearance Leakage Synthesis", fontsize=16)
    output_dir = project_path("runs/dals_vis")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dals_visualization.png"
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"saved -> {output_path}")


if __name__ == "__main__":
    main()
