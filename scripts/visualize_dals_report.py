"""汇报用 DALS（扩散外观渗漏合成）可视化：生成多张独立高清图。

相比 visualize_dals.py 的单张拼图，本脚本产出可直接放进 PPT/论文的分离图：
  fig1_pipeline.png      —— 5 步机制流程：采集病灶→二值足迹→热核扩散α→扩散外观→合成
  fig2_before_after.png  —— 原图 / 合成后 / 新增 lesion_2(红)，含放大框
  fig3_heat_profile.png  —— 热核径向剖面：硬贴图(阶跃) vs DALS软扩散(向外衰减)
  fig4_bank_montage.png  —— 实例库多样性网格(真实 lesion_2 crop + 轮廓)

用法：
    PYTHONPATH=src python scripts/visualize_dals_report.py
输出目录：runs/dals_vis/report/
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
from matplotlib.patches import ConnectionPatch, Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bs.augmentations import denormalize
from bs.dataset import UveitisSegmentationDataset, discover_samples
from bs.leakage_synthesis import (
    LeakageCopyPaste,
    build_instance_bank,
    gaussian_blur,
    leakage_instance_quality,
)
from bs.postprocess import _component_labels
from bs.paths import project_path


def to_disp(image: torch.Tensor) -> np.ndarray:
    return denormalize(image).clamp(0, 1).permute(1, 2, 0).numpy()


def lesion2(mask: torch.Tensor) -> np.ndarray:
    return ((mask == 2) | (mask == 3)).numpy()


def heat_kernel_maps(crop_image: torch.Tensor, crop_mask: torch.Tensor, sigma: float, gain: float):
    """复刻 LeakageCopyPaste._paste 的热核扩散步骤，返回 (alpha, appearance, density)。"""
    pad = int(round(3.0 * sigma))
    plane = F.pad(crop_mask.view(1, 1, *crop_mask.shape), (pad, pad, pad, pad)).squeeze(0)  # [1,H,W]
    image_plane = F.pad((crop_image * crop_mask.unsqueeze(0)).unsqueeze(0), (pad, pad, pad, pad)).squeeze(0)
    density = gaussian_blur(plane, sigma).clamp_min(1e-6)  # [1,H,W]
    appearance = (gaussian_blur(image_plane, sigma) / density).clamp(0.0, 1.0)  # [3,H,W]
    alpha = (density / density.max().clamp_min(1e-6)).clamp(0.0, 1.0)
    alpha = torch.maximum(alpha, plane).squeeze(0)  # [H,W]
    appearance = (appearance * gain).clamp(0.0, 1.0)
    return alpha, appearance, density.squeeze(0)


def pick_bright_instances(bank, k: int, min_area: float = 0.0, max_area: float = 1e9, dedup: bool = False):
    """按质量分+平均强度挑选清晰明亮的实例，便于汇报展示。

    dedup=True 时按 (面积, 平均强度) 签名去除近重复（数据集含 _aug 离线增强副本）。
    """
    scored = []
    for inst in bank:
        area = float(inst.mask.sum())
        if area < min_area or area > max_area:
            continue
        q = leakage_instance_quality(inst.image, inst.mask)
        scored.append((q["mean_intensity"] + q["score"], area, q["mean_intensity"], inst))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    picks: list = []
    seen: set = set()
    for _, area, mean_intensity, inst in scored:
        if dedup:
            sig = (int(area), round(float(mean_intensity), 3))
            if sig in seen:
                continue
            seen.add(sig)
        picks.append(inst)
        if len(picks) >= k:
            break
    return picks


# --------------------------------------------------------------------------------------
# Figure 1: 5 步机制流程图
# --------------------------------------------------------------------------------------
def fig_pipeline(bank, out_dir: Path) -> None:
    # 选一个面积适中、明亮的实例，视觉上最清楚
    candidates = pick_bright_instances(bank, 20)
    candidates.sort(key=lambda i: abs(float(i.mask.sum()) - 1500.0))
    inst = candidates[0]
    sigma, gain = 9.0, 1.35
    alpha, appearance, _ = heat_kernel_maps(inst.image, inst.mask, sigma, gain)

    # 合成到一块中性灰底 patch 上，突出“软融合”效果
    canvas = torch.full((3, *alpha.shape), 0.18)
    blended = (1.0 - alpha.unsqueeze(0)) * canvas + alpha.unsqueeze(0) * appearance
    blended = blended.clamp(0, 1).permute(1, 2, 0).numpy()

    crop_rgb = inst.image.clamp(0, 1).permute(1, 2, 0).numpy()
    footprint = inst.mask.numpy()

    titles = [
        "(1) Sampled real leak",
        "(2) Binary footprint",
        r"(3) Heat-kernel $\alpha$",
        "(4) Diffused appearance",
        "(5) Soft composite",
    ]
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.0))
    axes[0].imshow(crop_rgb)
    axes[1].imshow(footprint, cmap="gray")
    im = axes[2].imshow(alpha.numpy(), cmap="magma", vmin=0, vmax=1)
    axes[3].imshow((appearance * alpha.unsqueeze(0)).clamp(0, 1).permute(1, 2, 0).numpy())
    axes[4].imshow(blended)
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=13)
        ax.axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    # 面板间箭头
    for i in range(4):
        con = ConnectionPatch(
            xyA=(1.02, 0.5), coordsA=axes[i].transAxes,
            xyB=(-0.06, 0.5), coordsB=axes[i + 1].transAxes,
            arrowstyle="-|>", mutation_scale=22, lw=2.2, color="#333333",
        )
        fig.add_artist(con)

    fig.suptitle(
        "DALS Pipeline: real leak instance $\\rightarrow$ heat-kernel diffusion $\\rightarrow$ soft synthesis "
        "(bright core, diffuse concentration-decaying edge)",
        fontsize=14, y=1.04,
    )
    path = out_dir / "fig1_pipeline.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {path}")


# --------------------------------------------------------------------------------------
# Figure 2: 前后对比 + 放大框
# --------------------------------------------------------------------------------------
def fig_before_after(dataset, bank, out_dir: Path) -> None:
    block = LeakageCopyPaste(
        prob=1.0, instances=bank, targets=(2,), max_instances=(2, 3),
        scale=(0.9, 1.3), diffusion_sigma=(7.0, 12.0), intensity_gain=(1.3, 1.6),
        placement="macula_biased", alpha_threshold=0.5,
    )
    # 找一张原本无 lesion_2 的图
    target_idx = 0
    for index in range(len(dataset)):
        if lesion2(dataset[index]["mask"]).sum() == 0:
            target_idx = index
            break

    item = dataset[target_idx]
    before = to_disp(item["image"])
    before_l2 = lesion2(item["mask"])
    aug_img, aug_mask = block.apply(item["image"].clone(), item["mask"].clone())
    after = to_disp(aug_img)
    new_region = lesion2(aug_mask) & (~before_l2)
    overlay = after.copy()
    overlay[new_region] = [1.0, 0.15, 0.15]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    axes[0].imshow(before); axes[0].set_title(f"Original ({item['sample_id']})", fontsize=13)
    axes[1].imshow(after); axes[1].set_title("After DALS synthesis", fontsize=13)
    axes[2].imshow(overlay); axes[2].set_title(f"New lesion_2 mask (red)  +{int(new_region.sum())} px", fontsize=13)
    for ax in axes:
        ax.axis("off")

    # 对每个合成灶单独画框（散布放置时单一大框不准）
    h, w = new_region.shape
    for comp in _component_labels(new_region, 8):
        if len(comp) < 40:
            continue
        ys_c, xs_c = zip(*comp)
        pad = 18
        y0, y1 = max(0, min(ys_c) - pad), min(h, max(ys_c) + pad)
        x0, x1 = max(0, min(xs_c) - pad), min(w, max(xs_c) + pad)
        for ax in axes:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="yellow", lw=1.8))

    fig.suptitle("DALS enriches the extremely rare lesion_2 with realistic diffuse leakage", fontsize=14, y=1.0)
    path = out_dir / "fig2_before_after.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {path}")


# --------------------------------------------------------------------------------------
# Figure 3: 热核径向剖面 (硬贴图 vs 软扩散)
# --------------------------------------------------------------------------------------
def fig_heat_profile(bank, out_dir: Path) -> None:
    inst = pick_bright_instances(bank, 30)[0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # 左：不同 sigma 的软 alpha 剖面
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, 3))
    for sigma, color in zip([4.0, 8.0, 12.0], colors):
        alpha, _, _ = heat_kernel_maps(inst.image, inst.mask, sigma, 1.0)
        alpha_np = alpha.numpy()
        cy, cx = np.array(np.nonzero(alpha_np > 0.99)).mean(axis=1)
        yy, xx = np.mgrid[0 : alpha_np.shape[0], 0 : alpha_np.shape[1]]
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
        max_r = int(dist.max())
        radial = np.array([alpha_np[dist == r].mean() if (dist == r).any() else 0.0 for r in range(max_r + 1)])
        axes[0].plot(range(max_r + 1), radial, color=color, lw=2.4, label=f"DALS soft ($\\sigma$={sigma:g})")

    # 硬拷贝-粘贴的阶跃剖面作对照
    footprint = inst.mask.numpy()
    cy, cx = np.array(np.nonzero(footprint > 0.5)).mean(axis=1)
    yy, xx = np.mgrid[0 : footprint.shape[0], 0 : footprint.shape[1]]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    max_r = int(dist.max())
    hard = np.array([footprint[dist == r].mean() if (dist == r).any() else 0.0 for r in range(max_r + 1)])
    axes[0].plot(range(max_r + 1), hard, color="#d62728", lw=2.4, ls="--", label="Hard copy-paste")
    axes[0].set_xlabel("Radial distance from center (px)", fontsize=12)
    axes[0].set_ylabel(r"Blending weight $\alpha$", fontsize=12)
    axes[0].set_title("Concentration decays outward (soft) vs sharp edge (hard)", fontsize=12)
    axes[0].set_ylim(-0.03, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    # 右：热核 alpha 热力图
    alpha, _, _ = heat_kernel_maps(inst.image, inst.mask, 9.0, 1.0)
    im = axes[1].imshow(alpha.numpy(), cmap="magma", vmin=0, vmax=1)
    axes[1].set_title(r"Heat-kernel soft $\alpha$ ($\sigma$=9)", fontsize=12)
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle("DALS heat-kernel diffusion: physically-motivated soft leakage boundary", fontsize=14, y=1.0)
    path = out_dir / "fig3_heat_profile.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {path}")


# --------------------------------------------------------------------------------------
# Figure 4: 实例库多样性网格
# --------------------------------------------------------------------------------------
def fig_bank_montage(bank, out_dir: Path, rows: int = 3, cols: int = 6) -> None:
    picks = pick_bright_instances(bank, rows * cols, min_area=150.0, max_area=6000.0, dedup=True)
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.6 * rows))
    axes = np.atleast_2d(axes)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i >= len(picks):
            continue
        inst = picks[i]
        ax.imshow(inst.image.clamp(0, 1).permute(1, 2, 0).numpy())
        ax.contour(inst.mask.numpy(), levels=[0.5], colors="cyan", linewidths=1.2)
        ax.set_title(f"{int(inst.mask.sum())} px", fontsize=9)
    fig.suptitle(f"DALS instance bank (real lesion_2 crops, N={len(bank)}) — diverse shapes & scales", fontsize=14, y=1.0)
    path = out_dir / "fig4_bank_montage.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {path}")


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

    out_dir = project_path("runs/dals_vis/report")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_pipeline(bank, out_dir)
    fig_before_after(dataset, bank, out_dir)
    fig_heat_profile(bank, out_dir)
    fig_bank_montage(bank, out_dir)
    print(f"\nDone! 4 figures -> {out_dir}")


if __name__ == "__main__":
    main()
