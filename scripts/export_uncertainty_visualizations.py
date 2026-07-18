"""导出 UGI 三联可视化：原图 / 预测 overlay / TTA 不确定性热力图。

用法示例：
    python scripts/export_uncertainty_visualizations.py \
        --config configs/dinov3_convnext_tiny_diffleak.yaml \
        --checkpoint runs/<run>/f1/checkpoints/best.pt \
        --fold f1 --max-samples 12 --adr \
        --output-dir runs/<run>/f1/uncertainty_vis

需要 checkpoint 与已解压数据集；无训练开销，纯推理侧。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bs.convnext_seg import DinoV3ConvNeXtSegmentationModel
from bs.dataset import UveitisSegmentationDataset, discover_samples
from bs.model import DinoV3FpnSegmentationModel, DinoV3SegmentationModel
from bs.paths import project_path
from bs.uncertainty import anisotropic_diffusion_refine, make_triptych, tta_uncertainty


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export UGI uncertainty triptych visualizations.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", default="f1")
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", default=None, help="逗号分隔的 per-lesion 阈值，缺省用 config metric.threshold")
    parser.add_argument("--adr", action="store_true", help="启用各向异性扩散细化 (ADR)")
    parser.add_argument("--adr-iters", type=int, default=10)
    parser.add_argument("--adr-kappa", type=float, default=0.05)
    parser.add_argument("--adr-gamma", type=float, default=0.2)
    parser.add_argument("--no-tta", action="store_true", help="关闭 TTA (不确定性图将为 0)")
    return parser.parse_args()


def build_model(config: dict) -> torch.nn.Module:
    model_cfg = config["model"]
    train_cfg = config["train"]
    backbone = str(model_cfg["backbone"])
    if backbone.startswith("dinov3_convnext_"):
        return DinoV3ConvNeXtSegmentationModel(
            dinov3_code_dir=project_path(model_cfg["dinov3_code_dir"]),
            weights_path=project_path(model_cfg["backbone_weights"]),
            variant=str(model_cfg["variant"]),
            decoder_channels=int(model_cfg["decoder_channels"]),
            freeze_backbone=False,
            decoder_attention=str(model_cfg.get("decoder_attention", "none")),
            decoder_attention_reduction=int(model_cfg.get("decoder_attention_reduction", 16)),
            decoder_deep_supervision=False,
        )
    if backbone == "dinov3_vitb16":
        common = dict(
            dinov3_code_dir=project_path(model_cfg["dinov3_code_dir"]),
            weights_path=project_path(model_cfg["backbone_weights"]),
            intermediate_layers=list(model_cfg["intermediate_layers"]),
            num_classes=int(model_cfg.get("num_outputs", 2)),
            embed_dim=int(model_cfg["embed_dim"]),
            decoder_channels=int(model_cfg["decoder_channels"]),
            dropout=float(model_cfg.get("dropout", 0.1)),
            freeze_backbone=False,
            unfreeze_last_blocks=int(train_cfg.get("unfreeze_last_blocks", 0)),
        )
        if str(model_cfg.get("head", "token_fpn")) == "vit_fpn":
            return DinoV3FpnSegmentationModel(
                deep_supervision=bool(model_cfg.get("deep_supervision", True)),
                aux_loss_weight=float(model_cfg.get("aux_loss_weight", 0.4)),
                **common,
            )
        return DinoV3SegmentationModel(**common)
    raise ValueError(f"Unsupported backbone: {backbone}")


def resolve_thresholds(config: dict, override: str | None) -> list[float]:
    if override:
        values = [float(x) for x in override.split(",")]
    else:
        raw = config.get("metric", {}).get("threshold", 0.5)
        values = [float(x) for x in raw] if isinstance(raw, (list, tuple)) else [float(raw)]
    if len(values) == 1:
        values = values * 2
    return values


def resolve_tta(config: dict, disabled: bool) -> dict:
    if disabled:
        return {"enabled": False}
    tta = config.get("metric", {}).get("tta")
    if tta and bool(tta.get("enabled", False)):
        return tta
    return {"enabled": True, "flips": ["h", "v", "hv"], "scales": [1.0], "size_multiple": 32}


def main() -> None:
    args = parse_args()
    with project_path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()

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
    )
    dataset = UveitisSegmentationDataset(
        samples=samples,
        image_size=tuple(config["train"]["image_size"]),
        label_values=data_cfg["label_values"],
        ignore_index=data_cfg["ignore_index"],
        augment=False,
        preprocess_config=config.get("preprocess"),
    )

    thresholds = torch.tensor(resolve_thresholds(config, args.threshold)).view(2, 1, 1)
    tta_cfg = resolve_tta(config, args.no_tta)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    limit = min(int(args.max_samples), len(dataset))
    with torch.no_grad():
        for index in range(limit):
            item = dataset[index]
            image = item["image"].unsqueeze(0).to(device)
            mean_prob, uncertainty = tta_uncertainty(model, image, tta_cfg)
            if args.adr:
                mean_prob = anisotropic_diffusion_refine(
                    mean_prob, image, num_iters=args.adr_iters, kappa=args.adr_kappa, gamma=args.adr_gamma
                )
            pred = (mean_prob[0].cpu() >= thresholds).float()
            triptych = make_triptych(item["image"], pred, uncertainty[0].cpu())
            Image.fromarray(triptych).save(output_dir / f"{item['sample_id']}_triptych.png")
            print(f"saved {item['sample_id']}_triptych.png  (max_uncertainty={float(uncertainty.max()):.4f})")

    print(f"done: {limit} triptychs -> {output_dir}")


if __name__ == "__main__":
    main()
