"""对比 RDH-PDE (固定 Perona-Malik) 与 S3RD (Mamba) 的分割效果并挑出 badcase。

对某折验证集逐图用两个 checkpoint 预测，计算 per-lesion Dice，按"S3RD 在稀有类 lesion_2
上落后 RDH-PDE 最多"排序取 top-k，拼图 [原图 | GT | RDH-PDE | S3RD] 供 badcase 分析。

用法：
    python scripts/compare_rdh_s3rd.py \
        --pde-checkpoint runs/diffleak_f1_rdh_only/f1/checkpoints/best.pt \
        --ssm-checkpoint runs/diffleak_f1_s3rd/f1/checkpoints/best.pt \
        --fold f1 --topk 6 --threshold 0.5,0.5
输出：runs/compare_rdh_s3rd/compare.png
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bs.augmentations import denormalize
from bs.convnext_seg import DinoV3ConvNeXtSegmentationModel
from bs.dataset import UveitisSegmentationDataset, discover_samples
from bs.paths import project_path

_L1_COLOR = np.array([255, 64, 64], dtype=np.float32)
_L2_COLOR = np.array([255, 214, 0], dtype=np.float32)


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
        decoder_deep_supervision=False,
        head_type=str(m.get("head", "conv")),
        rdh_iters=int(rdh.get("iters", 8)),
        rdh_dt=float(rdh.get("dt", 0.2)),
        rdh_reaction=str(rdh.get("reaction", "fisher")),
        rdh_use_image_conductance=bool(rdh.get("use_image_conductance", True)),
        rdh_lambda=float(rdh.get("lambda", 0.1)),
        rdh_rho=float(rdh.get("rho", 1.0)),
        rdh_kappa=float(rdh.get("kappa", 0.1)),
        rdh_dynamics=str(rdh.get("dynamics", "pde")),
        rdh_d_state=int(rdh.get("d_state", 16)),
        rdh_directions=int(rdh.get("directions", 4)),
        rdh_stride=int(rdh.get("stride", 4)),
        rdh_d_inner=int(rdh.get("d_inner", 64)),
    )


def load_model(checkpoint_path: str, device: torch.device) -> DinoV3ConvNeXtSegmentationModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.no_grad()
def predict(model, image: torch.Tensor, thr: torch.Tensor) -> np.ndarray:
    prob = torch.sigmoid(model(image))[0].cpu()  # [2,H,W]
    return (prob >= thr.view(2, 1, 1)).numpy()


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return float("nan")  # 该通道无 GT 也无预测，跳过统计
    return float(2.0 * (pred & gt).sum() / denom)


def overlay(base: np.ndarray, l1: np.ndarray, l2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    out = base.astype(np.float32).copy()
    out[l1] = (1 - alpha) * out[l1] + alpha * _L1_COLOR
    out[l2] = (1 - alpha) * out[l2] + alpha * _L2_COLOR
    return out.clip(0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RDH-PDE vs S3RD and pick badcases.")
    parser.add_argument("--pde-checkpoint", required=True)
    parser.add_argument("--ssm-checkpoint", required=True)
    parser.add_argument("--fold", default="f1")
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--threshold", default="0.5,0.5")
    parser.add_argument("--output-dir", default="runs/compare_rdh_s3rd")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pde_model = load_model(args.pde_checkpoint, device)
    ssm_model = load_model(args.ssm_checkpoint, device)
    thr = torch.tensor([float(x) for x in args.threshold.split(",")])

    config = torch.load(args.pde_checkpoint, map_location="cpu", weights_only=False)["config"]
    data = config["data"]
    samples = discover_samples(
        dataset_root=project_path(data["root"]), folds=[args.fold],
        image_dir=data["image_dir"], mask_dir=data["mask_dir"],
        hrnet_result_dir=data.get("hrnet_result_dir", "HRNet_Result"),
        image_extensions=data["image_extensions"], mask_extensions=data["mask_extensions"],
        result_extensions=data.get("result_extensions", data["image_extensions"]),
        exclude_augmented=True,
    )
    dataset = UveitisSegmentationDataset(
        samples=samples, image_size=tuple(config["train"]["image_size"]),
        label_values=data["label_values"], ignore_index=data["ignore_index"], augment=False,
    )

    records = []
    d1_pde, d1_ssm, d2_pde, d2_ssm = [], [], [], []
    for idx in range(len(dataset)):
        item = dataset[idx]
        image = item["image"].unsqueeze(0).to(device)
        mask = item["mask"]
        gt1 = ((mask == 1) | (mask == 3)).numpy()
        gt2 = ((mask == 2) | (mask == 3)).numpy()
        pde = predict(pde_model, image, thr)
        ssm = predict(ssm_model, image, thr)
        rec = {
            "idx": idx, "id": item["sample_id"], "has_l2": bool(gt2.sum() > 0),
            "pde_d1": dice(pde[0], gt1), "pde_d2": dice(pde[1], gt2),
            "ssm_d1": dice(ssm[0], gt1), "ssm_d2": dice(ssm[1], gt2),
        }
        records.append(rec)
        if not np.isnan(rec["pde_d1"]): d1_pde.append(rec["pde_d1"]); d1_ssm.append(rec["ssm_d1"])
        if rec["has_l2"]: d2_pde.append(rec["pde_d2"]); d2_ssm.append(rec["ssm_d2"])

    print(f"val samples={len(records)}  has_lesion2={sum(r['has_l2'] for r in records)}")
    print(f"mean dice_1  PDE={np.mean(d1_pde):.4f}  SSM={np.mean(d1_ssm):.4f}")
    print(f"mean dice_2  PDE={np.mean(d2_pde):.4f}  SSM={np.mean(d2_ssm):.4f}  (仅含lesion_2的图)")

    # badcase: 在含 lesion_2 的图里, S3RD 的 dice_2 落后 RDH-PDE 最多
    cand = [r for r in records if r["has_l2"]]
    cand.sort(key=lambda r: (r["pde_d2"] - r["ssm_d2"]), reverse=True)
    picked = cand[: args.topk]

    rows = len(picked)
    fig, axes = plt.subplots(rows, 4, figsize=(16, 4.0 * rows))
    if rows == 1:
        axes = axes[None, :]
    for row, rec in enumerate(picked):
        item = dataset[rec["idx"]]
        base = (denormalize(item["image"]).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask = item["mask"]
        gt1 = ((mask == 1) | (mask == 3)).numpy(); gt2 = ((mask == 2) | (mask == 3)).numpy()
        image = item["image"].unsqueeze(0).to(device)
        pde = predict(pde_model, image, thr); ssm = predict(ssm_model, image, thr)
        axes[row, 0].imshow(base); axes[row, 0].set_title(f"Original ({rec['id']})")
        axes[row, 1].imshow(overlay(base, gt1, gt2)); axes[row, 1].set_title("GT (l1 red / l2 yellow)")
        axes[row, 2].imshow(overlay(base, pde[0], pde[1])); axes[row, 2].set_title(f"RDH-PDE  d2={rec['pde_d2']:.3f}")
        axes[row, 3].imshow(overlay(base, ssm[0], ssm[1])); axes[row, 3].set_title(f"S3RD(Mamba)  d2={rec['ssm_d2']:.3f}")
        for col in range(4):
            axes[row, col].axis("off")

    fig.suptitle("Badcase: RDH-PDE vs S3RD (S3RD lesion_2 落后最多的样本)", fontsize=15)
    output_dir = project_path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "compare.png"
    fig.tight_layout(rect=(0, 0, 1, 0.98)); fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"saved -> {path}")
    print("Top badcase (id, pde_d2, ssm_d2, gap):")
    for rec in picked:
        print(f"  {rec['id']}  pde={rec['pde_d2']:.3f}  ssm={rec['ssm_d2']:.3f}  gap={rec['pde_d2']-rec['ssm_d2']:.3f}")


if __name__ == "__main__":
    main()
