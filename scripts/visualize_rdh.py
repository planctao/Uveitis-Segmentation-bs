"""可视化 RDH（反应-扩散演化头）的内在可解释过程。

对若干验证图导出并拼图：
- 每图一行：原图 / 种子 s / 传导 c / 最终 u_K(overlay)
- 末行：某图的反应-扩散演化序列 u_0 -> u_K

体现 interpretable-by-design：分割是"从种子出发、受图像传导约束的扩散过程"的解。
用法：
    PYTHONPATH=src python scripts/visualize_rdh.py \
        --checkpoint runs/diffleak_f1_rdh/f1/checkpoints/best.pt --fold f1 --channel 0
输出：runs/rdh_vis/rdh_visualization.png
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
    parser = argparse.ArgumentParser(description="Visualize RDH reaction-diffusion evolution.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", default="f1")
    parser.add_argument("--channel", type=int, default=0, help="0=lesion_1, 1=lesion_2")
    parser.add_argument("--num-samples", type=int, default=2)
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
    n = min(int(args.num_samples), len(dataset))
    fig, axes = plt.subplots(n + 1, 4, figsize=(16, 4.2 * (n + 1)))

    first_steps = None
    for row in range(n):
        item = dataset[row]
        image = item["image"].unsqueeze(0).to(device)
        ev = rdh_evolution(model, image)
        seed = ev["seed"][0, ch].cpu().numpy()
        cond = ev["conductance"][0, ch].cpu().numpy()
        final_map = ev["final"][0, ch]
        if first_steps is None:
            first_steps = ev["steps"][:, 0, ch].cpu().numpy()

        base = to_disp(item["image"])
        final_up = (
            F.interpolate(final_map[None, None].float(), size=base.shape[:2], mode="bilinear", align_corners=False)[0, 0]
            .cpu()
            .numpy()
        )
        overlay = base.copy()
        overlay[final_up > 0.5] = [1.0, 0.0, 0.0]
        axes[row, 0].imshow(base); axes[row, 0].set_title(f"Original ({item['sample_id']})")
        axes[row, 1].imshow(seed, cmap="magma", vmin=0, vmax=1); axes[row, 1].set_title("Seed s (leak source)")
        axes[row, 2].imshow(cond, cmap="viridis", vmin=0, vmax=1); axes[row, 2].set_title("Conductance c (spread channel)")
        axes[row, 3].imshow(overlay); axes[row, 3].set_title("Final u_K (red)")
        for col in range(4):
            axes[row, col].axis("off")

    # 末行：演化序列 4 帧
    K = first_steps.shape[0]
    frames = sorted(set([0, K // 3, 2 * K // 3, K - 1]))
    for col in range(4):
        if col < len(frames):
            t = frames[col]
            axes[n, col].imshow(first_steps[t], cmap="magma", vmin=0, vmax=1)
            axes[n, col].set_title(f"Evolution u_{t}")
        axes[n, col].axis("off")

    fig.suptitle(f"RDH: Reaction-Diffusion Evolution (channel {ch})", fontsize=16)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rdh_visualization.png"
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"saved -> {output_path}")


if __name__ == "__main__":
    main()
