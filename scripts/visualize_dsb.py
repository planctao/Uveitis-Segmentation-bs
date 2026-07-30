"""汇报用 DSB（Diffusion Soft Boundary，扩散软边界监督）可视化。

DSB 在损失层对 GT 标签做处理：仅在病灶边界带内，用热核(高斯)扩散把硬 0/1 标签
软化为软标签，核心与远背景仍保持硬 0/1。动机：FA 渗漏边界本身模糊弥散，用软标签
监督比硬边界更贴合物理，缓解极小病灶(lesion_2)在边界处的过度惩罚。

本脚本直接调用 bs.multilabel.AsymmetricFocalTverskyBCE 的内部方法，作用于真实掩码，
无需训练模型。产出可直接放进 PPT/论文的分离图：
  fig1_dsb_pipeline.png —— 硬标签→边界带→热核扩散→DSB软目标 (含病灶区放大)
  fig2_dsb_profile.png  —— 跨边界 1D 剖面：硬阶跃 vs DSB软斜坡 + 2D软目标热图

用法：
    PYTHONPATH=src python scripts/visualize_dsb.py --channel 0
输出目录：runs/dsb_vis/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bs.augmentations import denormalize
from bs.dataset import UveitisSegmentationDataset, discover_samples
from bs.multilabel import AsymmetricFocalTverskyBCE, masks_to_paper_targets
from bs.paths import project_path
from bs.postprocess import _component_labels

# 与 diffleak_f1_dsb 训练一致的 DSB 参数
DSB_SIGMA = 2.0
DSB_BAND = 7
DSB_WEIGHT = 1.0


def make_loss(sigma: float, band: int = DSB_BAND, weight: float = DSB_WEIGHT) -> AsymmetricFocalTverskyBCE:
    return AsymmetricFocalTverskyBCE(
        soft_boundary_sigma=sigma, soft_boundary_band=band, soft_boundary_weight=weight
    )


def dsb_maps(mask: torch.Tensor, ch: int, sigma: float = DSB_SIGMA):
    """返回指定通道的 (hard, band, diffused, dsb_soft)。mask 为 [H,W] long。"""
    target, valid = masks_to_paper_targets(mask.unsqueeze(0))  # [1,2,H,W], [1,1,H,W]
    loss = make_loss(sigma)
    valid_f = valid.float()
    hard = target[0, ch]
    band = loss._boundary_band(target.float(), valid_f, loss.soft_boundary_band)[0, ch]
    diffused = loss._diffuse(target.float()).clamp(0.0, 1.0)[0, ch]
    dsb = loss._soft_boundary_target(target, valid.expand_as(target))[0, ch]
    return hard, band, diffused, dsb


def largest_component_centroid(binary: np.ndarray) -> tuple[int, int, tuple[int, int, int, int]]:
    comps = _component_labels(binary > 0.5, 8)
    if not comps:
        h, w = binary.shape
        return h // 2, w // 2, (0, 0, h - 1, w - 1)
    comp = max(comps, key=len)
    ys, xs = zip(*comp)
    cy, cx = int(np.mean(ys)), int(np.mean(xs))
    return cy, cx, (min(ys), min(xs), max(ys), max(xs))


def zoom_window(cy: int, cx: int, size: int, h: int, w: int) -> tuple[int, int, int, int]:
    half = size // 2
    y0 = int(np.clip(cy - half, 0, max(0, h - size)))
    x0 = int(np.clip(cx - half, 0, max(0, w - size)))
    return y0, x0, min(h, y0 + size), min(w, x0 + size)


def pick_sample(dataset, ch: int) -> int:
    """挑选目标通道病灶面积适中的样本，边界最清晰。"""
    best_idx, best_area = 0, -1.0
    for index in range(len(dataset)):
        mask = dataset[index]["mask"]
        area = float(((mask == (1 if ch == 0 else 2)) | (mask == 3)).sum())
        # 面积适中(2000~20000)优先，太大太散或太小都不利于展示
        score = area if 2000 <= area <= 20000 else area * 0.05
        if score > best_area:
            best_area, best_idx = score, index
    return best_idx


def fig_pipeline(image: torch.Tensor, hard, band, diffused, dsb, win, ch: int, out_dir: Path) -> None:
    y0, x0, y1, x1 = win
    base = denormalize(image).clamp(0, 1).permute(1, 2, 0).numpy()

    def crop(a):
        return a[y0:y1, x0:x1]

    fig, axes = plt.subplots(1, 5, figsize=(19, 4.2))
    axes[0].imshow(crop(base))
    axes[0].contour(crop(hard.numpy()), levels=[0.5], colors="cyan", linewidths=1.5)
    axes[0].set_title("(1) Image + lesion contour", fontsize=12)

    axes[1].imshow(crop(hard.numpy()), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("(2) Hard label (0/1)", fontsize=12)

    axes[2].imshow(crop(band.numpy()), cmap="cividis", vmin=0, vmax=1)
    axes[2].set_title(f"(3) Boundary band (k={DSB_BAND})", fontsize=12)

    im3 = axes[3].imshow(crop(diffused.numpy()), cmap="magma", vmin=0, vmax=1)
    axes[3].set_title(f"(4) Heat-kernel diffused ($\\sigma$={DSB_SIGMA:g})", fontsize=12)

    im4 = axes[4].imshow(crop(dsb.numpy()), cmap="magma", vmin=0, vmax=1)
    axes[4].set_title("(5) DSB soft target", fontsize=12)
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)

    lesion_name = "lesion_1" if ch == 0 else "lesion_2"
    fig.suptitle(
        f"DSB: Diffusion Soft Boundary supervision ({lesion_name})  —  "
        "hard core kept, only boundary softened by heat-kernel diffusion",
        fontsize=14, y=1.03,
    )
    path = out_dir / "fig1_dsb_pipeline.png"
    fig.savefig(path, dpi=165, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {path}")


def fig_profile(mask: torch.Tensor, hard, dsb, win, ch: int, out_dir: Path) -> None:
    y0, x0, y1, x1 = win
    hard_np, dsb_np = hard.numpy(), dsb.numpy()
    # 用最大连通域质心所在行做水平扫描线
    cy, cx, _ = largest_component_centroid(hard_np)
    cy = int(np.clip(cy, y0, y1 - 1))
    xs = np.arange(x0, x1)

    # 额外画一个更大 sigma 的软目标做“更明显”的示意
    _, _, _, dsb_big = dsb_maps(mask, ch, sigma=4.0)
    dsb_big_np = dsb_big.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].step(xs, hard_np[cy, x0:x1], where="mid", color="#1f77b4", lw=2.4, label="Hard label (step)")
    axes[0].plot(xs, dsb_np[cy, x0:x1], color="#ff7f0e", lw=2.4, label=f"DSB soft ($\\sigma$={DSB_SIGMA:g}, training)")
    axes[0].plot(xs, dsb_big_np[cy, x0:x1], color="#2ca02c", lw=2.0, ls="--", label="DSB soft ($\\sigma$=4, illustrative)")
    axes[0].set_xlabel("x (pixel) along scan line", fontsize=12)
    axes[0].set_ylabel("Supervision target value", fontsize=12)
    axes[0].set_title("Cross-boundary profile: sharp step $\\rightarrow$ smooth ramp", fontsize=12)
    axes[0].set_ylim(-0.05, 1.08)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10, loc="upper right")

    im = axes[1].imshow(dsb_np[y0:y1, x0:x1], cmap="magma", vmin=0, vmax=1)
    axes[1].axhline(cy - y0, color="cyan", lw=1.5, ls=":")
    axes[1].set_title(f"DSB soft target (zoom), scan line = cyan", fontsize=12)
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    lesion_name = "lesion_1" if ch == 0 else "lesion_2"
    fig.suptitle(f"DSB softens only the ambiguous FA leakage boundary ({lesion_name})", fontsize=14, y=1.0)
    path = out_dir / "fig2_dsb_profile.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize DSB (Diffusion Soft Boundary) supervision.")
    parser.add_argument("--fold", default="f1")
    parser.add_argument("--channel", type=int, default=0, help="0=lesion_1, 1=lesion_2")
    parser.add_argument("--sample-idx", type=int, default=-1, help="-1=auto pick a clear sample")
    parser.add_argument("--zoom", type=int, default=180, help="zoom window size (px)")
    parser.add_argument("--output-dir", default="runs/dsb_vis")
    args = parser.parse_args()

    root = project_path("dataset/dataset/split_dataorigin")
    samples = discover_samples(root, [args.fold], image_dir="img", mask_dir="mask_only_itksnap")
    dataset = UveitisSegmentationDataset(
        samples=samples, image_size=(768, 768), label_values=(0, 1, 2, 3), ignore_index=255, augment=False
    )

    ch = int(args.channel)
    idx = int(args.sample_idx) if args.sample_idx >= 0 else pick_sample(dataset, ch)
    item = dataset[idx]
    image, mask = item["image"], item["mask"]
    print(f"sample idx={idx} id={item['sample_id']} channel={ch}")

    hard, band, diffused, dsb = dsb_maps(mask, ch, sigma=DSB_SIGMA)
    cy, cx, _ = largest_component_centroid(hard.numpy())
    win = zoom_window(cy, cx, int(args.zoom), hard.shape[0], hard.shape[1])

    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_pipeline(image, hard, band, diffused, dsb, win, ch, out_dir)
    fig_profile(mask, hard, dsb, win, ch, out_dir)
    print(f"\nDone! 2 figures -> {out_dir}")


if __name__ == "__main__":
    main()
