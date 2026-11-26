"""Compare PBE's evaluation precision after a complete real FA training epoch.

Short architecture smoke checks cannot reveal running-statistic amplification.
This diagnostic keeps the formal screening configuration, saves the trained
state before evaluation, and reports the first non-finite module and FP32
comparison. The diagnostic is never included in performance comparisons.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_unet_multilabel import (
    build_loader, build_loss, build_model, build_optimizer, load_config,
    run_epoch, set_seed, setup_logger,
)


def inspect_forward(model, images, masks, criterion, amp):
    events, handles = [], []

    def inspect(name):
        def hook(module, inputs, output):
            if not isinstance(output, torch.Tensor):
                return
            value = output.detach().float()
            finite = torch.isfinite(value)
            events.append({"module": name, "dtype": str(output.dtype),
                           "finite": bool(finite.all()),
                           "nonfinite": int((~finite).sum()),
                           "max_abs": float(value[finite].abs().max()) if finite.any() else None})
        return hook

    for name, module in model.named_modules():
        if name and not list(module.children()):
            handles.append(module.register_forward_hook(inspect(name)))
    model.eval()
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp):
            output = model(images)
            loss = criterion(output, masks)
        return {"amp": amp, "loss": float(loss) if torch.isfinite(loss) else None,
                "output_finite": bool(torch.isfinite(output).all()),
                "first_nonfinite": next((event for event in events if not event["finite"]), None),
                "largest_finite": sorted((event for event in events if event["finite"]),
                                          key=lambda event: event["max_abs"], reverse=True)[:12],
                "module_events": events}
    finally:
        for handle in handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper_reproduction/pbeunet.yaml")
    parser.add_argument("--output", default="outputs/paper_reproduction_20261011/pbe_numerical_audit")
    parser.add_argument("--checkpoint", help="Reuse an already captured end-of-training-epoch state")
    parser.add_argument("--full-validation", action="store_true", help="Verify every clean validation batch")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    config["train"]["folds_to_run"] = ["f1"]
    set_seed(config["project"]["seed"])
    logger = setup_logger(output / "train.log", "pbe_numerical_audit")
    device = torch.device("cuda")
    train = build_loader(config, "f1", "train", logger)
    val = build_loader(config, "f1", "val", logger)
    model = build_model(config).to(device)
    criterion = build_loss(config)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model"])
        train_metrics = None
    else:
        optimizer = build_optimizer(model, config)
        scaler = torch.amp.GradScaler("cuda", enabled=config["train"]["amp"])
        with SummaryWriter(str(output / "tensorboard")) as writer:
            train_metrics = run_epoch(model, train, criterion, device, config, 1, logger, writer, optimizer, scaler)
        torch.save({"model": model.state_dict(), "config": config, "train_metrics": train_metrics},
                   output / "after_train.pt")
    batch = next(iter(val))
    images, masks = batch["image"].to(device), batch["mask"].to(device)
    report = {"experiment_class": "numerical_diagnostic_not_performance", "train_metrics": train_metrics,
              "state_nonfinite": [name for name, tensor in model.state_dict().items()
                                  if not torch.isfinite(tensor).all()],
              "validation_first_batch": [inspect_forward(model, images, masks, criterion, amp) for amp in [True, False]]}
    if args.full_validation:
        with SummaryWriter(str(output / "validation_tensorboard")) as writer:
            report["full_clean_validation"] = run_epoch(model, val, criterion, device, config, 1, logger, writer)
        report["validation_samples"] = len(val.dataset)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for result in report["validation_first_batch"]:
        print(json.dumps({key: value for key, value in result.items() if key != "module_events"}), flush=True)


if __name__ == "__main__":
    main()
