"""生成 RDH 反应-扩散演化的连续序列可视化。

输出一张水平时间轴图：u_0(seed) → u_1 → ... → u_K，
叠加在原图上，直观展示"渗漏从种子点出发、受传导约束逐步扩散"的物理过程。

用法：
    PYTHONPATH=src python scripts/visualize_rdh_evolution_sequence.py \
        --checkpoint runs/diffleak_f1_rdh_clean/f1/checkpoints/best.pt \
        --fold f1 --channel 0 --sample-idx 5
输出：runs/rdh_vis/rdh_evolution_sequence.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bs.augmentations import denormalize
from bs.convnext_seg import DinoV3ConvNeXtSegmentationModel
from bs.dataset import UveitisSegmentationDataset, discover_samples
from bs.paths import project_path


def build_model(config: dict) -> DinoV3ConvNeXtSegmentationModel:
    m = config["model"]
    rdh = m.get("rdh", {}) or {}
    return DinoV3ConvNeXtSegmentationModel(
        dinov3_code_dir=project_path(m["dinov3_code_dir"]),
        weights_path=project_path(m["backbone_weights"]),
        variant=str(m.get("variant", "tiny")),
        decoder_channels=int(m.get("decoder_channels", 192)),
        freeze_backbone=False,
        decoder_attention=str(m.get("decoder_attention", "none")),
        decoder_attention_reduction=int(m.get("decoder_attention_reduction", 16)),
        decoder_deep_supervision=False,
        head_type=str(m.get("head", "conv")),
        rdh_iters=int(rdh.get("iters", 8)),
        rdh_dt=float(rdh.get("dt", 0.2)),
        rdh_reaction=str(rdh.get("reaction", "fisher")),
        rdh_use_image_conductance=bool(rdh.get("use_image_conductance", True)),
        rdh_lambda=float(rdh.get("lambda", 0.1)),
        rdh_rho=float(rdh.get("rho", 1.0)),
        rdh_kappa=float(rdh.get("kappa", 0.1)),
    )


@torch.no_grad()
def rdh_evolution(model: DinoV3ConvNeXtSegmentationModel, images: torch.Tensor) -> dict:
    features = model.extract_multiscale_features(images)
    dec = model.decode_head
    pyramid = [layer(feature) for layer, feature in zip(dec.lateral, features)]
    for idx in range(len(pyramid) - 1, 0, -1):
        up = F.interpolate(pyramid[idx], size=pyramid[idx - 1].shape[-2:], mode="bilinear", align_corners=False)
        pyramid[idx - 1] = dec.smooth[idx - 1](pyramid[idx - 1] + up)
    target = pyramid[0].shape[-2:]
    fused = torch.cat(
        [f if f.shape[-2:] == target else F.interpolate(f, size=target, mode="bilinear", align_corners=False) for f in pyramid],
        dim=1,
    )
    fused = dec.attention(fused)
    feat = dec.neck(fused)
    guide = (
        F.interpolate(images, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        if dec.rdh_head.use_image_conductance
        else None
    )
    return dec.rdh_head.evolution(feat, guide)


def to_disp(image: torch.Tensor) -> np.ndarray:
    return denormalize(image).clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="RDH evolution sequence visualization.")
    parser.add_argument("--checkpoint", default="runs/diffleak_f1_rdh_clean/f1/checkpoints/best.pt")
    parser.add_argument("--fold", default="f1")
    parser.add_argument("--channel", type=int, default=0, help="0=lesion_1, 1=lesion_2")
    parser.add_argument("--sample-idx", type=int, default=0, help="dataset sample index")
    parser.add_argument("--output-dir", default="runs/rdh_vis")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if str(config["model"].get("head", "conv")) != "rdh":
        raise SystemExit("该 checkpoint 不是 RDH 头，无法可视化演化过程")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    data = config["data"]
    samples = discover_samples(
        dataset_root=project_path(data["root"]),
        folds=[args.fold],
        image_dir=data["image_dir"],
        mask_dir=data["mask_dir"],
        hrnet_result_dir=data.get("hrnet_result_dir", "HRNet_Result"),
        image_extensions=data["image_extensions"],
        mask_extensions=data["mask_extensions"],
        result_extensions=data.get("result_extensions", data["image_extensions"]),
    )
    dataset = UveitisSegmentationDataset(
        samples=samples,
        image_size=tuple(config["train"]["image_size"]),
        label_values=data["label_values"],
        ignore_index=data["ignore_index"],
        augment=False,
        preprocess_config=config.get("preprocess"),
    )

    ch = int(args.channel)
    idx = min(int(args.sample_idx), len(dataset) - 1)
    item = dataset[idx]
    image = item["image"].unsqueeze(0).to(device)
    ev = rdh_evolution(model, image)

    # steps: [iters+1, B, out_ch, H, W]
    steps = ev["steps"][:, 0, ch].cpu().numpy()  # [K+1, H, W]
    conductance = ev["conductance"][0, ch].cpu().numpy()
    base = to_disp(item["image"])
    K = steps.shape[0] - 1  # 总步数

    # 上采样到原图尺寸
    H, W = base.shape[:2]
    steps_up = np.stack([
        F.interpolate(
            torch.from_numpy(steps[t])[None, None].float(),
            size=(H, W), mode="bilinear", align_corners=False
        )[0, 0].numpy()
        for t in range(K + 1)
    ])
    cond_up = F.interpolate(
        torch.from_numpy(conductance)[None, None].float(),
        size=(H, W), mode="bilinear", align_corners=False
    )[0, 0].numpy()

    # ---- 图1: 完整演化序列 (每步一帧) ----
    n_frames = K + 1
    fig, axes = plt.subplots(1, n_frames, figsize=(3.2 * n_frames, 3.6))
    if n_frames == 1:
        axes = [axes]

    for t in range(n_frames):
        axes[t].imshow(base)
        # 用半透明热力图叠加
        mask = steps_up[t]
        axes[t].imshow(mask, cmap="inferno", alpha=0.6, vmin=0, vmax=1)
        axes[t].set_title(f"$u_{{{t}}}$", fontsize=13)
        axes[t].axis("off")

    fig.suptitle(
        f"RDH Reaction-Diffusion Evolution  (channel={ch}, sample={item['sample_id']})\n"
        r"$u_{t+1} = u_t + \Delta t\,[\,\mathrm{div}(c\,\nabla u_t) + \rho\, s\, u_t(1-u_t) - \lambda\, u_t\,]$",
        fontsize=12, y=1.02,
    )
    output_dir = Path(project_path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    path1 = output_dir / "rdh_evolution_sequence.png"
    fig.tight_layout()
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[1/3] saved -> {path1}")

    # ---- 图2: 关键帧对比 (seed / 中间 / 最终 + 传导图) ----
    key_frames = sorted(set([0, K // 4, K // 2, 3 * K // 4, K]))
    n_key = len(key_frames) + 1  # +1 for conductance
    fig2, axes2 = plt.subplots(1, n_key, figsize=(3.5 * n_key, 4.0))

    for i, t in enumerate(key_frames):
        axes2[i].imshow(base)
        axes2[i].imshow(steps_up[t], cmap="inferno", alpha=0.6, vmin=0, vmax=1)
        label = "seed $u_0=s$" if t == 0 else f"$u_{{{t}}}$"
        if t == K:
            label = f"final $u_{{{K}}}$"
        axes2[i].set_title(label, fontsize=12)
        axes2[i].axis("off")

    # 传导系数
    axes2[-1].imshow(base)
    axes2[-1].imshow(cond_up, cmap="cool", alpha=0.5, vmin=0, vmax=1)
    axes2[-1].set_title("Conductance $c$", fontsize=12)
    axes2[-1].axis("off")

    fig2.suptitle("RDH Key Frames + Conductance", fontsize=13, y=1.01)
    path2 = output_dir / "rdh_evolution_keyframes.png"
    fig2.tight_layout()
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[2/3] saved -> {path2}")

    # ---- 图3: 单像素扩散曲线 (选最大种子点) ----
    peak_loc = np.unravel_index(steps_up[0].argmax(), steps_up[0].shape)
    values_at_peak = [steps_up[t][peak_loc] for t in range(K + 1)]

    # 也画几个邻域点
    offsets = [(0, 0), (10, 0), (20, 0), (40, 0), (80, 0)]
    fig3, ax3 = plt.subplots(figsize=(7, 4.5))
    for dy, dx in offsets:
        py, px = peak_loc[0] + dy, peak_loc[1] + dx
        if 0 <= py < H and 0 <= px < W:
            vals = [steps_up[t][py, px] for t in range(K + 1)]
            dist = int(np.sqrt(dy**2 + dx**2))
            ax3.plot(range(K + 1), vals, marker="o", markersize=4, label=f"dist={dist}px")

    ax3.set_xlabel("Evolution step $t$", fontsize=12)
    ax3.set_ylabel("Activation $u_t$", fontsize=12)
    ax3.set_title("Diffusion Propagation at Different Distances from Seed Peak", fontsize=12)
    ax3.set_ylim(-0.05, 1.05)
    ax3.set_xticks(range(K + 1))
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    path3 = output_dir / "rdh_diffusion_curve.png"
    fig3.tight_layout()
    fig3.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"[3/3] saved -> {path3}")

    print(f"\nDone! 共 {K} 步演化, 输出目录: {output_dir}")


if __name__ == "__main__":
    main()
