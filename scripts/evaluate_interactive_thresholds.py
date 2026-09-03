"""Evaluate an interactive-refiner checkpoint with lesion-specific threshold sweep.

The training/evaluation loop historically reports PaperDice at threshold 0.5.
This utility keeps the click simulator fixed at the training threshold, then
searches independent output thresholds for lesion_1 and lesion_2.  It is an
offline calibration diagnostic and does not alter the checkpoint.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_interactive_refiner import (  # noqa: E402
    build_loader,
    build_model,
    get_feature_builder,
    load_config,
    make_inputs,
    resolve_click_radius,
    resolve_config,
    seed_everything,
)
from bs.interactive_refiner import (  # noqa: E402
    _controlled_residual_update,
    oracle_refine_with_target,
)
from bs.interactive_refiner import enforce_click_constraints  # noqa: E402
from bs.multilabel import masks_to_paper_targets  # noqa: E402
from bs.paths import project_path  # noqa: E402
from bs.postprocess import apply_postprocessor, build_postprocessor  # noqa: E402
from bs.adaptive_threshold import build_threshold_adapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dino_sam_refiner.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", choices=["f1", "f2", "f3", "f4", "f5"], required=True)
    parser.add_argument("--clicks", default="0,1,3,5")
    parser.add_argument("--thresholds", default="0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--strategy", default=None, choices=["random", "center", "farthest"])
    parser.add_argument("--mode", default=None, choices=["disk", "gaussian"])
    parser.add_argument(
        "--radius",
        default=None,
        help="Prompt radius, either a scalar or comma-separated per-lesion values (e.g. 12,8).",
    )
    parser.add_argument(
        "--residual-gate",
        default=None,
        choices=["none", "uncertainty", "uncertainty_click", "uncertainty_click_channelwise"],
    )
    parser.add_argument("--gate-floor", type=float, default=None)
    parser.add_argument("--gate-click-gain", type=float, default=None)
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=None,
        help="Parameter-free residual multiplier for checkpoint calibration.",
    )
    parser.add_argument(
        "--residual-channel-scale",
        default=None,
        help="Optional scalar or comma-separated per-lesion residual scales (e.g. 1.0,0.8).",
    )
    parser.add_argument(
        "--residual-step-limit",
        type=float,
        default=None,
        help="Optional tanh bound on each residual update (logit units).",
    )
    parser.add_argument(
        "--total-residual-limit",
        type=float,
        default=None,
        help="Optional tanh bound on cumulative rollout drift from the initial DINO logits.",
    )
    parser.add_argument(
        "--click-constraint-strength",
        type=float,
        default=None,
        help="Optional non-learned projection strength for honoring click polarity.",
    )
    parser.add_argument(
        "--click-jitter",
        type=float,
        default=None,
        help="Optional Gaussian click-location jitter in pixels for robustness evaluation.",
    )
    parser.add_argument(
        "--click-dropout",
        type=float,
        default=None,
        help="Optional probability of dropping each simulated click for robustness evaluation.",
    )
    parser.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Average original/horizontal/vertical/diagonal flips (one-shot only).",
    )
    parser.add_argument(
        "--disable-postprocess",
        action="store_true",
        help="Ignore an optional postprocess section from the evaluation config.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = resolve_config(load_config(args.config), args)
    config.setdefault("train", {})["folds_to_run"] = [args.fold]
    config["train"]["batch_size"] = int(args.batch_size)
    config.setdefault("runtime", {})["num_workers"] = int(args.num_workers)
    if args.strategy is not None:
        config.setdefault("clicks", {})["strategy"] = args.strategy
    if args.mode is not None:
        config.setdefault("clicks", {})["mode"] = args.mode
    if args.radius is not None:
        config.setdefault("clicks", {})["radius"] = resolve_click_radius(args.radius)
    if args.click_jitter is not None:
        if args.click_jitter < 0.0:
            raise ValueError("--click-jitter must be non-negative")
        config.setdefault("clicks", {})["jitter"] = float(args.click_jitter)
    if args.click_dropout is not None:
        if not 0.0 <= args.click_dropout <= 1.0:
            raise ValueError("--click-dropout must be in [0, 1]")
        config.setdefault("clicks", {})["dropout"] = float(args.click_dropout)

    checkpoint_path = project_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint.get("config")
    if isinstance(checkpoint_config, dict):
        # Preserve checkpoint model/click settings while allowing the explicit
        # fold/device/loader choices above.
        merged = dict(checkpoint_config)
        # Evaluation-only sections (for example morphology postprocessing)
        # are intentionally absent from the training checkpoint.  Keep them
        # from the requested evaluation config instead of dropping them when
        # the checkpoint configuration is merged.
        for key, value in config.items():
            merged.setdefault(key, value)
        for key in ("runtime", "train", "data", "cache", "clicks"):
            if key in config:
                merged[key] = config[key]
        config = merged
    # These are parameter-free model behavior switches, so they can be
    # evaluated against an existing checkpoint without retraining.
    if args.residual_gate is not None:
        config.setdefault("model", {})["residual_gate"] = args.residual_gate
    if args.gate_floor is not None:
        config.setdefault("model", {})["residual_gate_floor"] = float(args.gate_floor)
    if args.gate_click_gain is not None:
        config.setdefault("model", {})["residual_gate_click_gain"] = float(args.gate_click_gain)
    if args.residual_scale is not None:
        if args.residual_scale < 0.0:
            raise ValueError("--residual-scale must be non-negative")
        config.setdefault("model", {})["residual_scale"] = float(args.residual_scale)
    residual_channel_scale = None
    if args.residual_channel_scale is not None:
        values = [float(item.strip()) for item in args.residual_channel_scale.split(",") if item.strip()]
        if not values:
            raise ValueError("--residual-channel-scale must contain at least one value")
        if any(value < 0.0 for value in values):
            raise ValueError("--residual-channel-scale values must be non-negative")
        residual_channel_scale = values[0] if len(values) == 1 else values
    if args.residual_step_limit is not None and args.residual_step_limit < 0.0:
        raise ValueError("--residual-step-limit must be non-negative")
    if args.tta and bool(config.get("clicks", {}).get("iterative", False)):
        raise ValueError("--tta currently supports one-shot MVP evaluation only")
    if args.total_residual_limit is not None and args.total_residual_limit < 0.0:
        raise ValueError("--total-residual-limit must be non-negative")

    # CLI values override the YAML, while a positive YAML value remains active
    # when the flag is omitted.  Keeping the effective values explicit avoids
    # silently evaluating a named stable configuration as an unconstrained
    # rollout.
    raw_step_limit = (
        args.residual_step_limit
        if args.residual_step_limit is not None
        else config.get("clicks", {}).get("residual_step_limit")
    )
    raw_total_limit = (
        args.total_residual_limit
        if args.total_residual_limit is not None
        else config.get("clicks", {}).get("total_residual_limit")
    )
    if raw_step_limit is not None and float(raw_step_limit) < 0.0:
        raise ValueError("residual_step_limit must be non-negative")
    if raw_total_limit is not None and float(raw_total_limit) < 0.0:
        raise ValueError("total_residual_limit must be non-negative")
    effective_step_limit = None if raw_step_limit is None or float(raw_step_limit) <= 0.0 else float(raw_step_limit)
    effective_total_limit = None if raw_total_limit is None or float(raw_total_limit) <= 0.0 else float(raw_total_limit)

    device_name = config.get("runtime", {}).get("device", "cuda")
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    loader = build_loader(config, args.fold, "val")
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    model.eval()
    postprocessor = None if args.disable_postprocess else build_postprocessor(config.get("postprocess"))
    threshold_adapter = build_threshold_adapter(config.get("adaptive_threshold"))

    clicks = [int(x.strip()) for x in args.clicks.split(",") if x.strip()]
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    pairs = list(itertools.product(thresholds, thresholds))
    results: dict[str, dict[str, object]] = {}
    with torch.no_grad():
        for count in clicks:
            # Reset the simulator RNG per click count so model/threshold
            # variants are compared on identical oracle point samples.
            seed_everything(int(args.seed) + int(count))
            intersections = torch.zeros(len(pairs), 2, dtype=torch.float64)
            predicted = torch.zeros_like(intersections)
            targets = torch.zeros_like(intersections)
            for batch in loader:
                batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
                target, valid = masks_to_paper_targets(batch["mask"], int(config["data"].get("ignore_index", 255)))
                if bool(config.get("clicks", {}).get("iterative", False)):
                    # Strict SAM-style closed loop: each round observes the
                    # latest refiner prediction before selecting the next
                    # oracle error click.
                    result = oracle_refine_with_target(
                        model,
                        batch["image"],
                        batch["dino_logits"],
                        target,
                        count,
                        threshold=float(config["clicks"].get("threshold", 0.5)),
                        radius=resolve_click_radius(config["clicks"].get("radius", 8)),
                        mode=str(config["clicks"].get("mode", "disk")),
                        strategy=str(config["clicks"].get("strategy", "random")),
                        feature_builder=get_feature_builder(config),
                        click_jitter=float(config.get("clicks", {}).get("jitter", 0.0) or 0.0),
                        click_dropout=float(config.get("clicks", {}).get("dropout", 0.0) or 0.0),
                        residual_step_limit=effective_step_limit,
                        total_residual_limit=effective_total_limit,
                        stop_gradient=False,
                        teacher_forcing_ratio=0.0,
                        residual_channel_scale=residual_channel_scale,
                    )
                    logits = result.logits
                    if args.click_constraint_strength is not None:
                        logits = enforce_click_constraints(
                            logits,
                            result.positive_clicks,
                            result.negative_clicks,
                            strength=float(args.click_constraint_strength),
                        )
                else:
                    # Original MVP semantics: sample all points once, render
                    # cumulative heatmaps, and run one residual update.
                    features, _, _, positive_clicks, negative_clicks = make_inputs(
                        batch, config, count, return_prompts=True
                    )
                    if args.tta:
                        # Flip the full prompt-conditioned feature tensor so
                        # the exact same clicks are used in every orientation.
                        # Each prediction is inverse-flipped before averaging;
                        # this is a zero-parameter test-time ensemble.
                        augmented: list[Tensor] = []
                        for dims in (None, (-1,), (-2,), (-2, -1)):
                            aug_features = features if dims is None else torch.flip(features, dims=dims)
                            aug_dino = (
                                batch["dino_logits"]
                                if dims is None
                                else torch.flip(batch["dino_logits"], dims=dims)
                            )
                            aug_logits = _controlled_residual_update(
                                model,
                                aug_features,
                                aug_dino,
                                residual_step_limit=effective_step_limit,
                                anchor_state=aug_dino,
                                total_residual_limit=effective_total_limit,
                                residual_channel_scale=residual_channel_scale,
                            )
                            if dims is not None:
                                aug_logits = torch.flip(aug_logits, dims=dims)
                            augmented.append(aug_logits)
                        logits = torch.stack(augmented, dim=0).mean(dim=0)
                    else:
                        logits = _controlled_residual_update(
                            model,
                            features,
                            batch["dino_logits"],
                            residual_step_limit=effective_step_limit,
                            anchor_state=batch["dino_logits"],
                            total_residual_limit=effective_total_limit,
                            residual_channel_scale=residual_channel_scale,
                        )
                    if args.click_constraint_strength is not None:
                        logits = enforce_click_constraints(
                            logits,
                            positive_clicks,
                            negative_clicks,
                            strength=float(args.click_constraint_strength),
                        )
                probs = torch.sigmoid(logits).cpu()
                target = target.cpu().bool()
                valid = valid.cpu().bool().expand_as(target)
                for idx, (t1, t2) in enumerate(pairs):
                    if threshold_adapter is None:
                        thresholds_tensor = torch.tensor([t1, t2], dtype=probs.dtype).view(1, 2, 1, 1)
                    else:
                        thresholds_tensor = threshold_adapter(probs, [t1, t2])
                    pred = probs >= thresholds_tensor
                    if postprocessor is not None:
                        pred = apply_postprocessor(pred, postprocessor, probabilities=probs)
                    pred = pred & valid
                    gt = target & valid
                    intersections[idx] += (pred & gt).sum(dim=(0, 2, 3)).double()
                    predicted[idx] += pred.sum(dim=(0, 2, 3)).double()
                    targets[idx] += gt.sum(dim=(0, 2, 3)).double()
            dice = 2.0 * intersections / (predicted + targets).clamp_min(1.0)
            macro = dice.mean(dim=1)
            best = int(macro.argmax())
            results[str(count)] = {
                "best_threshold": list(pairs[best]),
                "dice_1": float(dice[best, 0]),
                "dice_2": float(dice[best, 1]),
                "macro_dice": float(macro[best]),
                "grid": [
                    {"threshold": list(pairs[i]), "dice_1": float(dice[i, 0]), "dice_2": float(dice[i, 1]), "macro_dice": float(macro[i])}
                    for i in range(len(pairs))
                ],
            }
    payload = {
        "fold": args.fold,
        "checkpoint": str(checkpoint_path),
        "seed": int(args.seed),
        "clicks": clicks,
        "thresholds": thresholds,
        "strategy": str(config.get("clicks", {}).get("strategy", "random")),
        "radius": config.get("clicks", {}).get("radius", 8),
        "mode": str(config.get("clicks", {}).get("mode", "disk")),
        "iterative": bool(config.get("clicks", {}).get("iterative", False)),
        "tta": bool(args.tta),
        "residual_gate": str(config.get("model", {}).get("residual_gate", "none")),
        "gate_floor": float(config.get("model", {}).get("residual_gate_floor", 0.25)),
        "gate_click_gain": float(config.get("model", {}).get("residual_gate_click_gain", 0.75)),
        "residual_scale": float(config.get("model", {}).get("residual_scale", 1.0)),
        "residual_channel_scale": residual_channel_scale,
        "residual_step_limit": effective_step_limit,
        "total_residual_limit": effective_total_limit,
        "click_constraint_strength": args.click_constraint_strength,
        "click_jitter": float(config.get("clicks", {}).get("jitter", 0.0) or 0.0),
        "click_dropout": float(config.get("clicks", {}).get("dropout", 0.0) or 0.0),
        "postprocess": config.get("postprocess") if postprocessor is not None else None,
        "adaptive_threshold": config.get("adaptive_threshold") if threshold_adapter is not None else None,
        "results": results,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
