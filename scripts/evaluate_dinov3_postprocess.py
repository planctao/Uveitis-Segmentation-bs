from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bs.convnext_seg import DinoV3ConvNeXtSegmentationModel
from bs.adaptive_threshold import build_threshold_adapter
from bs.fov import apply_fov_mask, build_fov_masker
from bs.intensity_refine import apply_intensity_refiner, build_intensity_refiner
from bs.model import DinoV3FpnSegmentationModel, DinoV3SegmentationModel
from bs.multilabel import PaperDice, masks_to_paper_targets
from bs.paths import project_path
from bs.postprocess import apply_postprocessor, build_postprocessor
from bs.tta import predict_with_tta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DINOv3 checkpoint with threshold, TTA, morphology, and FOV postprocess.")
    parser.add_argument("--config", default="configs/dinov3_convnext_tiny_multilabel_morph_tta.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", choices=["f1", "f2", "f3", "f4", "f5"], required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--threshold", default=None, help="Scalar or two comma-separated thresholds, e.g. 0.5 or 0.5,0.9")
    parser.add_argument("--tta-scales", default=None, help="Comma-separated scale TTA values, e.g. 0.875,1.0,1.125")
    parser.add_argument("--uncertainty-penalty", default=None, help="Scalar or two comma-separated UATTA penalties, e.g. 0.15 or 0.0,0.2")
    parser.add_argument("--tta-appearance-mode", choices=["none", "fa_lce"], default=None)
    parser.add_argument("--tta-appearance-strength", type=float, default=None)
    parser.add_argument("--tta-appearance-kernel", type=int, default=None)
    parser.add_argument("--tta-appearance-quantile", type=float, default=None)
    parser.add_argument("--tta-appearance-channel-reduce", choices=["max", "mean", "green"], default=None)
    parser.add_argument("--preprocess-mode", choices=["none", "fa_lce"], default=None)
    parser.add_argument("--preprocess-strength", type=float, default=None)
    parser.add_argument("--preprocess-kernel", type=int, default=None)
    parser.add_argument("--preprocess-quantile", type=float, default=None)
    parser.add_argument("--preprocess-channel-reduce", choices=["max", "mean", "green"], default=None)
    parser.add_argument("--disable-tta", action="store_true")
    parser.add_argument("--disable-postprocess", action="store_true")
    parser.add_argument("--disable-intensity-refine", action="store_true")
    parser.add_argument("--disable-fov-mask", action="store_true")
    parser.add_argument("--ablation-suite", action="store_true", help="Evaluate default, calibrated threshold, TTA, morphology, and FOV variants in one pass.")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with project_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_threshold(value: str | None, default: float | list[float]) -> float | list[float]:
    if value is None:
        return default
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts
    raise ValueError(f"Expected one or two thresholds, got {value}")


def parse_scalar_or_pair(value: str | None) -> float | list[float] | None:
    if value is None:
        return None
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts
    raise ValueError(f"Expected one or two values, got {value}")


def parse_float_list(value: str | None) -> list[float] | None:
    if value is None:
        return None
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError(f"Expected at least one value, got {value}")
    return values


def build_loader(config: dict[str, Any], fold: str) -> DataLoader:
    from bs.dataset import UveitisSegmentationDataset, discover_samples

    data_cfg = config["data"]
    train_cfg = config["train"]
    samples = discover_samples(
        dataset_root=project_path(data_cfg["root"]),
        folds=[fold],
        image_dir=data_cfg["image_dir"],
        mask_dir=data_cfg["mask_dir"],
        hrnet_result_dir=data_cfg.get("hrnet_result_dir", "HRNet_Result"),
        image_extensions=data_cfg["image_extensions"],
        mask_extensions=data_cfg["mask_extensions"],
        result_extensions=data_cfg.get("result_extensions", data_cfg["image_extensions"]),
    )
    if train_cfg.get("max_val_samples"):
        samples = samples[: int(train_cfg["max_val_samples"])]
    dataset = UveitisSegmentationDataset(
        samples=samples,
        image_size=tuple(train_cfg["image_size"]),
        label_values=data_cfg["label_values"],
        ignore_index=data_cfg["ignore_index"],
        augment=False,
        augmentation_config=None,
        preprocess_config=config.get("preprocess"),
    )
    return DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(config["runtime"]["num_workers"]),
        pin_memory=True,
        drop_last=False,
        persistent_workers=int(config["runtime"]["num_workers"]) > 0,
    )


def build_model(config: dict[str, Any]) -> nn.Module:
    model_cfg = config["model"]
    train_cfg = config["train"]
    backbone = str(model_cfg["backbone"])
    head_type = str(model_cfg.get("head", "token_fpn"))
    if backbone == "dinov3_vitb16":
        if head_type == "vit_fpn":
            return DinoV3FpnSegmentationModel(
                dinov3_code_dir=project_path(model_cfg["dinov3_code_dir"]),
                weights_path=project_path(model_cfg["backbone_weights"]),
                intermediate_layers=list(model_cfg["intermediate_layers"]),
                num_classes=int(model_cfg.get("num_outputs", 2)),
                embed_dim=int(model_cfg["embed_dim"]),
                decoder_channels=int(model_cfg["decoder_channels"]),
                dropout=float(model_cfg["dropout"]),
                freeze_backbone=bool(train_cfg["freeze_backbone"]),
                unfreeze_last_blocks=int(train_cfg.get("unfreeze_last_blocks", 0)),
                deep_supervision=bool(model_cfg.get("deep_supervision", True)),
                aux_loss_weight=float(model_cfg.get("aux_loss_weight", 0.4)),
            )
        wbe_cfg = config.get("wbe", {})
        return DinoV3SegmentationModel(
            dinov3_code_dir=project_path(model_cfg["dinov3_code_dir"]),
            weights_path=project_path(model_cfg["backbone_weights"]),
            intermediate_layers=list(model_cfg["intermediate_layers"]),
            num_classes=int(model_cfg.get("num_outputs", 2)),
            embed_dim=int(model_cfg["embed_dim"]),
            decoder_channels=int(model_cfg["decoder_channels"]),
            dropout=float(model_cfg["dropout"]),
            freeze_backbone=bool(train_cfg["freeze_backbone"]),
            unfreeze_last_blocks=int(train_cfg.get("unfreeze_last_blocks", 0)),
            use_wbe=bool(wbe_cfg.get("enabled", False)),
            wbe_shared=bool(wbe_cfg.get("shared", False)),
            wbe_reduction=int(wbe_cfg.get("reduction", 4)),
            wbe_bottleneck=int(wbe_cfg.get("bottleneck_channels", 256)),
            wbe_version=int(wbe_cfg.get("version", 1)),
            wbe_snr_temperature=float(wbe_cfg.get("snr_temperature", 1.0)),
        )
    if backbone.startswith("dinov3_convnext_"):
        return DinoV3ConvNeXtSegmentationModel(
            dinov3_code_dir=project_path(model_cfg["dinov3_code_dir"]),
            weights_path=project_path(model_cfg["backbone_weights"]),
            variant=str(model_cfg["variant"]),
            decoder_channels=int(model_cfg["decoder_channels"]),
            freeze_backbone=bool(train_cfg["freeze_backbone"]),
            decoder_attention=str(model_cfg.get("decoder_attention", "none")),
            decoder_attention_reduction=int(model_cfg.get("decoder_attention_reduction", 16)),
            decoder_deep_supervision=bool(model_cfg.get("decoder_deep_supervision", False)),
        )
    raise ValueError(f"Unsupported backbone: {backbone}")


class ThresholdSweepDice:
    def __init__(
        self,
        thresholds: list[float],
        ignore_index: int,
        postprocessor: Any = None,
        fov_masker: Any = None,
        intensity_refiner: Any = None,
    ) -> None:
        self.thresholds = torch.tensor(thresholds, dtype=torch.float64)
        self.ignore_index = int(ignore_index)
        self.postprocessor = postprocessor
        self.fov_masker = fov_masker
        self.intensity_refiner = intensity_refiner
        self.intersections = torch.zeros((len(thresholds), 2), dtype=torch.float64)
        self.predicted = torch.zeros((len(thresholds), 2), dtype=torch.float64)
        self.targets = torch.zeros((len(thresholds), 2), dtype=torch.float64)

    def update(self, logits: torch.Tensor, mask: torch.Tensor, image: torch.Tensor | None = None) -> None:
        target, valid = masks_to_paper_targets(mask.detach().cpu(), self.ignore_index)
        target = target.bool()
        valid = valid.expand_as(target).bool()
        probs = torch.sigmoid(logits.detach().cpu()).to(torch.float64)
        dims = (0, 2, 3)
        for idx, threshold in enumerate(self.thresholds):
            pred = (probs >= float(threshold.item())) & valid
            if self.postprocessor is not None:
                pred = apply_postprocessor(pred, self.postprocessor, probabilities=probs.float()) & valid
            if self.intensity_refiner is not None:
                if image is None:
                    raise ValueError("ThresholdSweepDice with intensity_refiner requires image in update(logits, mask, image)")
                pred = apply_intensity_refiner(pred, image.detach().cpu(), self.intensity_refiner, probabilities=probs.float()) & valid
            if self.fov_masker is not None:
                if image is None:
                    raise ValueError("ThresholdSweepDice with fov_masker requires image in update(logits, mask, image)")
                pred = apply_fov_mask(pred, image.detach().cpu(), self.fov_masker, probabilities=probs.float()) & valid
            tgt = target & valid
            self.intersections[idx] += (pred & tgt).sum(dim=dims).to(torch.float64)
            self.predicted[idx] += pred.sum(dim=dims).to(torch.float64)
            self.targets[idx] += tgt.sum(dim=dims).to(torch.float64)

    def compute(self) -> dict[str, float]:
        dice = (2.0 * self.intersections / (self.predicted + self.targets).clamp_min(1.0)).nan_to_num(0.0)
        macro = dice.mean(dim=1)
        best_idx = int(torch.argmax(macro).item())
        best_idx_1 = int(torch.argmax(dice[:, 0]).item())
        best_idx_2 = int(torch.argmax(dice[:, 1]).item())
        return {
            "sweep_best_threshold": float(self.thresholds[best_idx].item()),
            "sweep_best_dice_1": float(dice[best_idx, 0].item()),
            "sweep_best_dice_2": float(dice[best_idx, 1].item()),
            "sweep_best_macro_dice": float(macro[best_idx].item()),
            "sweep_ind_threshold_1": float(self.thresholds[best_idx_1].item()),
            "sweep_ind_threshold_2": float(self.thresholds[best_idx_2].item()),
            "sweep_ind_dice_1": float(dice[best_idx_1, 0].item()),
            "sweep_ind_dice_2": float(dice[best_idx_2, 1].item()),
            "sweep_ind_macro_dice": float((0.5 * (dice[best_idx_1, 0] + dice[best_idx_2, 1])).item()),
        }


def load_checkpoint(model: nn.Module, path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)


def build_eval_context(config: dict[str, Any], checkpoint: Path, fold: str) -> tuple[torch.device, DataLoader, nn.Module]:
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    loader = build_loader(config, fold)
    model = build_model(config).to(device)
    load_checkpoint(model, checkpoint)
    model.eval()
    return device, loader, model


def evaluate(config: dict[str, Any], checkpoint: Path, fold: str) -> dict[str, Any]:
    device, loader, model = build_eval_context(config, checkpoint, fold)

    metric_cfg = config.get("metric", {})
    postprocessor = build_postprocessor(metric_cfg.get("postprocess"))
    threshold_adapter = build_threshold_adapter(metric_cfg.get("adaptive_threshold"))
    fov_masker = build_fov_masker(metric_cfg.get("fov_mask"))
    intensity_refiner = build_intensity_refiner(metric_cfg.get("intensity_refine"))
    dice = PaperDice(
        ignore_index=int(config["data"]["ignore_index"]),
        threshold=metric_cfg["threshold"],
        postprocessor=postprocessor,
        fov_masker=fov_masker,
        intensity_refiner=intensity_refiner,
        threshold_adapter=threshold_adapter,
    )
    sweep_cfg = metric_cfg.get("threshold_sweep", {})
    sweep = None
    if bool(sweep_cfg.get("enabled", False)):
        sweep = ThresholdSweepDice(
            thresholds=[float(x) for x in sweep_cfg.get("thresholds", [])],
            ignore_index=int(config["data"]["ignore_index"]),
            postprocessor=postprocessor if bool(sweep_cfg.get("postprocess", False)) else None,
            fov_masker=fov_masker if bool(sweep_cfg.get("fov_mask", False)) else None,
            intensity_refiner=intensity_refiner if bool(sweep_cfg.get("intensity_refine", False)) else None,
        )

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"eval {fold}", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=bool(config["train"]["amp"]) and device.type == "cuda"):
                logits = predict_with_tta(model, images, metric_cfg.get("tta"))
            dice.update(logits, masks, images)
            if sweep is not None:
                sweep.update(logits, masks, images)

    result: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "fold": fold,
        "samples": len(loader.dataset),
        "threshold": metric_cfg["threshold"],
        "tta": metric_cfg.get("tta", {"enabled": False}),
        "adaptive_threshold": metric_cfg.get("adaptive_threshold", {"enabled": False}),
        "postprocess": metric_cfg.get("postprocess", {"enabled": False}),
        "intensity_refine": metric_cfg.get("intensity_refine", {"enabled": False}),
        "fov_mask": metric_cfg.get("fov_mask", {"enabled": False}),
        **dice.compute(),
    }
    if sweep is not None:
        result.update(sweep.compute())
    return result


def _metric_result(
    name: str,
    threshold: float | list[float],
    logits_source: str,
    postprocess_name: str,
    intensity_name: str,
    fov_name: str,
    metric: PaperDice,
) -> dict[str, Any]:
    return {
        "name": name,
        "threshold": threshold,
        "logits": logits_source,
        "postprocess": postprocess_name,
        "intensity_refine": intensity_name,
        "fov_mask": fov_name,
        **metric.compute(),
    }


def evaluate_ablation_suite(config: dict[str, Any], checkpoint: Path, fold: str) -> dict[str, Any]:
    device, loader, model = build_eval_context(config, checkpoint, fold)
    ignore_index = int(config["data"]["ignore_index"])
    metric_cfg = config.get("metric", {})
    calibrated_threshold = metric_cfg["threshold"]
    tta_cfg = metric_cfg.get("tta", {"enabled": False})
    postprocess_cfg = metric_cfg.get("postprocess", {"enabled": False})
    postprocessor = build_postprocessor(postprocess_cfg)
    postprocess_name = "morphology" if postprocessor is not None else "none"
    adaptive_threshold_cfg = metric_cfg.get("adaptive_threshold", {"enabled": False})
    threshold_adapter = build_threshold_adapter(adaptive_threshold_cfg)
    intensity_cfg = metric_cfg.get("intensity_refine", {"enabled": False})
    intensity_refiner = build_intensity_refiner(intensity_cfg)
    intensity_name = "fa_intensity" if intensity_refiner is not None else "none"
    fov_mask_cfg = metric_cfg.get("fov_mask", {"enabled": False})
    fov_masker = build_fov_masker(fov_mask_cfg)
    fov_name = "fov" if fov_masker is not None else "none"

    variants = {
        "default_0_5": {
            "metric": PaperDice(ignore_index=ignore_index, threshold=0.5),
            "logits": "base",
            "threshold": 0.5,
            "postprocess": "none",
            "intensity_refine": "none",
            "fov_mask": "none",
        },
        "calibrated_threshold": {
            "metric": PaperDice(ignore_index=ignore_index, threshold=calibrated_threshold),
            "logits": "base",
            "threshold": calibrated_threshold,
            "postprocess": "none",
            "intensity_refine": "none",
            "fov_mask": "none",
        },
        "calibrated_threshold_tta": {
            "metric": PaperDice(ignore_index=ignore_index, threshold=calibrated_threshold),
            "logits": "tta",
            "threshold": calibrated_threshold,
            "postprocess": "none",
            "intensity_refine": "none",
            "fov_mask": "none",
        },
        "calibrated_threshold_morph": {
            "metric": PaperDice(ignore_index=ignore_index, threshold=calibrated_threshold, postprocessor=postprocessor),
            "logits": "base",
            "threshold": calibrated_threshold,
            "postprocess": postprocess_name,
            "intensity_refine": "none",
            "fov_mask": "none",
        },
        "fcm_tta_full": {
            "metric": PaperDice(ignore_index=ignore_index, threshold=calibrated_threshold, postprocessor=postprocessor),
            "logits": "tta",
            "threshold": calibrated_threshold,
            "postprocess": postprocess_name,
            "intensity_refine": "none",
            "fov_mask": "none",
        },
    }
    if intensity_refiner is not None:
        variants["calibrated_threshold_intensity"] = {
            "metric": PaperDice(ignore_index=ignore_index, threshold=calibrated_threshold, intensity_refiner=intensity_refiner),
            "logits": "base",
            "threshold": calibrated_threshold,
            "postprocess": "none",
            "intensity_refine": intensity_name,
            "fov_mask": "none",
        }
        variants["fcm_tta_intensity_full"] = {
            "metric": PaperDice(
                ignore_index=ignore_index,
                threshold=calibrated_threshold,
                postprocessor=postprocessor,
                intensity_refiner=intensity_refiner,
            ),
            "logits": "tta",
            "threshold": calibrated_threshold,
            "postprocess": postprocess_name,
            "intensity_refine": intensity_name,
            "fov_mask": "none",
        }
    if fov_masker is not None:
        variants["calibrated_threshold_fov"] = {
            "metric": PaperDice(ignore_index=ignore_index, threshold=calibrated_threshold, fov_masker=fov_masker),
            "logits": "base",
            "threshold": calibrated_threshold,
            "postprocess": "none",
            "intensity_refine": "none",
            "fov_mask": fov_name,
        }
        variants["fcm_tta_fov_full"] = {
            "metric": PaperDice(
                ignore_index=ignore_index,
                threshold=calibrated_threshold,
                postprocessor=postprocessor,
                fov_masker=fov_masker,
            ),
            "logits": "tta",
            "threshold": calibrated_threshold,
            "postprocess": postprocess_name,
            "intensity_refine": "none",
            "fov_mask": fov_name,
        }
    if intensity_refiner is not None and fov_masker is not None:
        variants["fcm_tta_intensity_fov_full"] = {
            "metric": PaperDice(
                ignore_index=ignore_index,
                threshold=calibrated_threshold,
                postprocessor=postprocessor,
                intensity_refiner=intensity_refiner,
                fov_masker=fov_masker,
            ),
            "logits": "tta",
            "threshold": calibrated_threshold,
            "postprocess": postprocess_name,
            "intensity_refine": intensity_name,
            "fov_mask": fov_name,
        }
    if threshold_adapter is not None:
        variants["adaptive_threshold_tta"] = {
            "metric": PaperDice(ignore_index=ignore_index, threshold=calibrated_threshold, threshold_adapter=threshold_adapter),
            "logits": "tta",
            "threshold": calibrated_threshold,
            "postprocess": "none",
            "intensity_refine": "none",
            "fov_mask": "none",
        }
        variants["adaptive_threshold_tta_full"] = {
            "metric": PaperDice(
                ignore_index=ignore_index,
                threshold=calibrated_threshold,
                postprocessor=postprocessor,
                intensity_refiner=intensity_refiner,
                fov_masker=fov_masker,
                threshold_adapter=threshold_adapter,
            ),
            "logits": "tta",
            "threshold": calibrated_threshold,
            "postprocess": postprocess_name,
            "intensity_refine": intensity_name,
            "fov_mask": fov_name,
        }

    sweep_cfg = metric_cfg.get("threshold_sweep", {})
    sweep_base = None
    sweep_tta = None
    if bool(sweep_cfg.get("enabled", False)):
        thresholds = [float(x) for x in sweep_cfg.get("thresholds", [])]
        sweep_base = ThresholdSweepDice(thresholds=thresholds, ignore_index=ignore_index)
        sweep_tta = ThresholdSweepDice(thresholds=thresholds, ignore_index=ignore_index)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"ablation {fold}", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=bool(config["train"]["amp"]) and device.type == "cuda"):
                base_logits = predict_with_tta(model, images, {"enabled": False})
                tta_logits = predict_with_tta(model, images, tta_cfg)
            for variant in variants.values():
                logits = tta_logits if variant["logits"] == "tta" else base_logits
                variant["metric"].update(logits, masks, images)
            if sweep_base is not None and sweep_tta is not None:
                sweep_base.update(base_logits, masks, images)
                sweep_tta.update(tta_logits, masks, images)

    results = [
        _metric_result(
            name=name,
            threshold=variant["threshold"],
            logits_source=variant["logits"],
            postprocess_name=variant["postprocess"],
            intensity_name=variant["intensity_refine"],
            fov_name=variant["fov_mask"],
            metric=variant["metric"],
        )
        for name, variant in variants.items()
    ]
    best_variant = max(results, key=lambda row: float(row["paper_macro_dice"]))
    payload: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "fold": fold,
        "samples": len(loader.dataset),
        "tta": tta_cfg,
        "adaptive_threshold": adaptive_threshold_cfg,
        "postprocess": postprocess_cfg,
        "intensity_refine": intensity_cfg,
        "fov_mask": fov_mask_cfg,
        "variants": results,
        "best_variant": best_variant,
    }
    if sweep_base is not None and sweep_tta is not None:
        payload["raw_threshold_sweep_base"] = sweep_base.compute()
        payload["raw_threshold_sweep_tta"] = sweep_tta.compute()
    return payload


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["runtime"]["num_workers"] = args.num_workers
    config["metric"]["threshold"] = parse_threshold(args.threshold, config["metric"]["threshold"])
    if args.disable_tta:
        config["metric"]["tta"] = {"enabled": False}
    scales = parse_float_list(args.tta_scales)
    if scales is not None:
        config["metric"].setdefault("tta", {"enabled": True})
        config["metric"]["tta"]["scales"] = scales
    penalty = parse_scalar_or_pair(args.uncertainty_penalty)
    if penalty is not None:
        config["metric"].setdefault("tta", {"enabled": True})
        config["metric"]["tta"]["uncertainty_penalty"] = penalty
    if args.tta_appearance_mode is not None:
        tta = config["metric"].setdefault("tta", {"enabled": True})
        appearance = tta.setdefault("appearance_preprocess", {})
        if args.tta_appearance_mode == "none":
            appearance["enabled"] = False
        else:
            appearance["enabled"] = True
            appearance["mode"] = args.tta_appearance_mode
    appearance_overrides = {
        "strength": args.tta_appearance_strength,
        "kernel_size": args.tta_appearance_kernel,
        "quantile": args.tta_appearance_quantile,
        "channel_reduce": args.tta_appearance_channel_reduce,
    }
    if any(value is not None for value in appearance_overrides.values()):
        tta = config["metric"].setdefault("tta", {"enabled": True})
        appearance = tta.setdefault("appearance_preprocess", {"enabled": True, "mode": "fa_lce"})
        for key, value in appearance_overrides.items():
            if value is not None:
                appearance[key] = value
    if args.preprocess_mode is not None:
        preprocess = config.setdefault("preprocess", {})
        if args.preprocess_mode == "none":
            preprocess["enabled"] = False
        else:
            preprocess["enabled"] = True
            preprocess["mode"] = args.preprocess_mode
    preprocess_overrides = {
        "strength": args.preprocess_strength,
        "kernel_size": args.preprocess_kernel,
        "quantile": args.preprocess_quantile,
        "channel_reduce": args.preprocess_channel_reduce,
    }
    if any(value is not None for value in preprocess_overrides.values()):
        preprocess = config.setdefault("preprocess", {})
        for key, value in preprocess_overrides.items():
            if value is not None:
                preprocess[key] = value
    if args.disable_postprocess:
        config["metric"]["postprocess"] = {"enabled": False}
    if args.disable_intensity_refine:
        config["metric"]["intensity_refine"] = {"enabled": False}
    if args.disable_fov_mask:
        config["metric"]["fov_mask"] = {"enabled": False}

    if args.ablation_suite:
        result = evaluate_ablation_suite(config, project_path(args.checkpoint), args.fold)
    else:
        result = evaluate(config, project_path(args.checkpoint), args.fold)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        output_path = project_path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
