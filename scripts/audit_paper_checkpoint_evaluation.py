"""Re-evaluate a completed paper checkpoint on its complete clean FA fold.

Checks the saved selected checkpoint against reported Dice/loss using actual
images and masks. This is validation reproducibility, not an independent test.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_paper_reproduction import atomic_json
from scripts.train_unet_multilabel import set_seed, setup_logger
from scripts.fusion_evaluation_components import evaluation_components
from scripts.verify_paper_experiment import SUMMARY_FIELDS, verify_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--fold", default="f1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run, output = Path(args.run), Path(args.output)
    verified = verify_run(run, args.fold, require_full_data=True)
    config = json.loads((run / "resolved_config.json").read_text())
    build_loader, build_model, build_loss, run_epoch = evaluation_components(config)
    set_seed(int(config["project"]["seed"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output.with_suffix(".log"), "paper_checkpoint_evaluation")
    loader = build_loader(config, args.fold, "val", logger)
    device = torch.device(config["runtime"]["device"])
    model = build_model(config).to(device)
    checkpoint = torch.load(run / args.fold / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    criterion = build_loss(config)
    started = time.monotonic()
    with SummaryWriter(str(output.with_suffix(".tensorboard"))) as writer:
        measured = run_epoch(model, loader, criterion, device, config, verified["best_epoch"], logger, writer)
    differences = {key: abs(float(measured[field]) - float(verified["metrics"][key]))
                   for key, field in SUMMARY_FIELDS.items()}
    matching = all(math.isclose(float(measured[field]), float(verified["metrics"][key]), rel_tol=1e-6, abs_tol=1e-6)
                   for key, field in SUMMARY_FIELDS.items())
    report = {"run": str(run.resolve()), "fold": args.fold, "best_epoch": verified["best_epoch"],
              "checkpoint_sha256": verified["checkpoints"]["best"]["sha256"],
              "resolved_config_sha256": verified["resolved_config_sha256"],
              "metric_history_sha256": verified["metric_history_sha256"],
              "validation_samples": len(loader.dataset), "validation_augmented_excluded": True,
              "measured": measured, "reported": verified["metrics"], "absolute_differences": differences,
              "all_required_metrics_match": matching, "seconds": time.monotonic() - started,
              "scope": "full clean-fold re-evaluation of the selected checkpoint; same validation used for model and threshold selection"}
    atomic_json(output, report)
    print(json.dumps(report, indent=2, allow_nan=False))
    if not matching:
        raise RuntimeError(f"Checkpoint re-evaluation differs from reported metrics; inspect {output}")


if __name__ == "__main__":
    main()
