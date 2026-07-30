"""可视化 Oriented-PDC (CPDC/APDC/RPDC) 多方向差分卷积组的边缘响应。

用训练好的 edge(pdc bank) 模型跑真实 FA 验证图，展示:
  原图 | GT(lesion_1/2) | 分割概率(l1/l2) | 学习边缘图 | CPDC | APDC | RPDC
直观看出多方向算子的分工: 血管(取向条状)在 APDC 上亮, 渗漏(径向外扩)边界在 RPDC 上亮。

用法:
  PYTHONPATH=src python scripts/visualize_oriented_pdc.py \
    --checkpoint runs/edge_orientedpdc_f1/f1/checkpoints/best.pt \
    --fold f1 --num-samples 4 --output-dir runs/pdc_vis
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bs.convnext_seg import DinoV3ConvNeXtSegmentationModel
from bs.dataset import RGB_LABEL_COLORS, UveitisSegmentationDataset, decode_mask_array, discover_samples
from bs.paths import project_path

import nibabel as nib
from PIL import Image


def sample_lesion_flags(mask_path: str) -> tuple[bool, bool]:
    path = Path(mask_path)
    if path.name.lower().endswith((".nii.gz", ".nii")):
        array = np.asanyarray(nib.load(str(path)).dataobj)
    else:
        colors = Image.open(path).convert("RGB").getcolors(maxcolors=256)
        if colors is not None:
            labels = {RGB_LABEL_COLORS[color] for _, color in colors if color in RGB_LABEL_COLORS}
            if labels:
                return bool({1, 3} & labels), bool({2, 3} & labels)
        array = np.asarray(Image.open(path))
    array = decode_mask_array(array, path)
    return bool(np.any((array == 1) | (array == 3))), bool(np.any((array == 2) | (array == 3)))


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="runs/edge_orientedpdc_f1/f1/checkpoints/best.pt")
    p.add_argument("--fold", default="f1")
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--output-dir", default="runs/pdc_vis")
    p.add_argument("--pdc-types", default="cpdc,apdc,rpdc")
    return p.parse_args()


def pick_samples(fold: str, num: int) -> list:
    samples = discover_samples(
        dataset_root=project_path("dataset/dataset/split_dataorigin"),
        folds=[fold],
        image_dir="img",
        mask_dir="mask_only_itksnap",
        hrnet_result_dir="HRNet_Result",
        image_extensions=[".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"],
        mask_extensions=[".nii.gz", ".nii", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"],
        result_extensions=[".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"],
        exclude_augmented=True,
    )
    with_l2, with_l1 = [], []
    for s in samples:
        l1, l2 = sample_lesion_flags(str(s.mask_path))
        (with_l2 if l2 else with_l1 if l1 else []).append(s)
    take_l2 = with_l2[: max(1, num - 1)]
    chosen = take_l2 + with_l1[: num - len(take_l2)]
    return chosen[:num]


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pdc_types = [t.strip() for t in args.pdc_types.split(",") if t.strip()]

    model = DinoV3ConvNeXtSegmentationModel(
        dinov3_code_dir=str(PROJECT_ROOT / "backbone/dinov3"),
        weights_path=None,
        variant="tiny",
        decoder_channels=192,
        freeze_backbone=False,
        head_type="edge",
        edge_pdc_types=pdc_types,
    ).to(device)
    ckpt = torch.load(project_path(args.checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    branch_outputs: dict[int, torch.Tensor] = {}
    bank = model.decode_head.edge_head.edge_branch
    for i, branch in enumerate(bank.branches):
        branch.register_forward_hook(lambda m, inp, out, i=i: branch_outputs.__setitem__(i, out))

    samples = pick_samples(args.fold, args.num_samples)
    print(f"samples: {[s.image_path.name for s in samples]}")
    dataset = UveitisSegmentationDataset(
        samples=samples, image_size=(768, 768), label_values=[0, 1, 2, 3],
        ignore_index=255, augment=False, augmentation_config=None, preprocess_config=None,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, sample in enumerate(samples):
        batch = dataset[idx]
        image = batch["image"].unsqueeze(0).to(device)
        mask = batch["mask"]
        with torch.no_grad():
            decoder = model.decode_head
            features = model.extract_multiscale_features(image)
            pyramid = [layer(f) for layer, f in zip(decoder.lateral, features)]
            for i in range(len(pyramid) - 1, 0, -1):
                up = F.interpolate(pyramid[i], size=pyramid[i - 1].shape[-2:], mode="bilinear", align_corners=False)
                pyramid[i - 1] = decoder.smooth[i - 1](pyramid[i - 1] + up)
            target = pyramid[0].shape[-2:]
            fused = torch.cat(
                [f if f.shape[-2:] == target else F.interpolate(f, size=target, mode="bilinear", align_corners=False) for f in pyramid],
                dim=1,
            )
            fused = decoder.attention(fused)
            feat = decoder.neck(fused)
            seg_logits, edge_logits = decoder.edge_head(feat)

        probs = torch.sigmoid(F.interpolate(seg_logits, size=(768, 768), mode="bilinear", align_corners=False))[0].cpu()
        img_np = np.clip(image[0].cpu().numpy().transpose(1, 2, 0) * _IMAGENET_STD + _IMAGENET_MEAN, 0, 1)
        gray = img_np.max(axis=2)
        gt1 = ((mask == 1) | (mask == 3)).numpy()
        gt2 = ((mask == 2) | (mask == 3)).numpy()
        edge_np = torch.sigmoid(edge_logits[0]).mean(0).cpu().numpy()

        panels = [
            ("FA 原图", gray, "gray"),
            ("GT (l1红/l2蓝)", np.dstack([gt1.astype(np.float32), np.zeros_like(gray), gt2.astype(np.float32)]), None),
            ("Pred l1 概率", probs[0].numpy(), "hot"),
            ("Pred l2 概率", probs[1].numpy(), "hot"),
            ("学习边缘图", edge_np, "magma"),
        ]
        for name in pdc_types:
            resp = branch_outputs[pdc_types.index(name)][0].abs().mean(0).cpu().numpy()
            resp = (resp - resp.min()) / (np.ptp(resp) + 1e-8)
            panels.append((f"{name.upper()} 响应", resp, "inferno"))

        fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.3))
        for ax, (title, arr, cmap) in zip(axes, panels):
            if cmap is None:
                ax.imshow(arr)
            else:
                ax.imshow(arr, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.suptitle(sample.image_path.name, fontsize=11)
        fig.tight_layout()
        out_path = out_dir / f"pdc_vis_{idx}_{sample.image_path.stem}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
