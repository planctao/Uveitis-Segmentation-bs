"""Measure the real 768 FA forward/backward cost of the paired DINO decoders."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.run_paper_reproduction import atomic_json
from scripts import train_dinov3_multilabel as dino
from scripts import train_dino_fusion_paired as paired
from scripts.train_unet_multilabel import sample_lesion_flags, setup_logger
from bs.paired_experiments import initialize_paired_model
from bs.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/edg_validation_20261011")
    args = parser.parse_args()
    torch.set_num_threads(2)
    paired.configure_training()
    root = (PROJECT_ROOT / args.root).resolve()
    root.relative_to(PROJECT_ROOT / "outputs")
    output = root / "dino_768_memory_probe.json"
    logger = setup_logger(output.with_suffix(".log"), "dino_fusion_memory_probe")
    configs = {mode: yaml.safe_load((root / "pending_configs" / f"paired_dino_{mode}_vsubr_s42_f1_30_768.yaml").read_text())
               for mode in ("sum", "equal", "edg")}
    data_config = json.loads(json.dumps(configs["sum"]))
    data_config["runtime"]["num_workers"] = 0
    dataset = dino.build_loader(data_config, "f1", "val", logger).dataset
    positive, negative = None, None
    for index, sample in enumerate(dataset.samples):
        has_lesion2 = sample_lesion_flags(str(sample.mask_path))[1]
        if has_lesion2 and positive is None:
            positive = index
        if not has_lesion2 and negative is None:
            negative = index
        if positive is not None and negative is not None:
            break
    if positive is None or negative is None:
        raise RuntimeError("Probe requires a rare-lesion positive and negative image")
    samples = [dataset[index] for index in (positive, negative)]
    device = torch.device("cuda")
    results = {}
    for mode, config in configs.items():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = initialize_paired_model(paired.build_model, config).to(device).train()
        criterion = dino.build_loss(config)
        optimizer = dino.build_optimizer(model, config)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        images = torch.stack([sample["image"] for sample in samples]).to(device)
        masks = torch.stack([sample["mask"] for sample in samples]).to(device)
        attempts = []
        for attempt in range(1, 7):
            set_seed(42 + attempt)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, masks, image=images)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {mode} probe loss")
            scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["clip_grad_norm"])
            finite = bool(torch.isfinite(grad_norm))
            if finite:
                scaler.step(optimizer)
            scaler.update()
            attempts.append({"attempt": attempt, "amp_scale": scale, "loss": float(loss),
                             "finite_gradient_norm": finite, "gradient_norm": float(grad_norm) if finite else None})
            if finite:
                break
        results[mode] = {"resolution": [768, 768], "microbatch": len(samples),
                         "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
                         "finite_forward_and_eventual_backward": finite, "attempts": attempts}
        logger.info("%s probe %s", mode, json.dumps(results[mode]))
        atomic_json(output, {"status": "running", "results": results,
                             "sample_ids": [sample["sample_id"] for sample in samples],
                             "scope": "two real FA images; memory and AMP runtime probe only; not an accuracy experiment"})
        del model, criterion, optimizer, scaler, images, masks, logits, loss, grad_norm
        if not finite:
            raise FloatingPointError(f"{mode} gradients remained nonfinite after six AMP backoffs")
    report = {"status": "passed", "results": results, "sample_ids": [sample["sample_id"] for sample in samples],
              "scope": "two real FA images; one lesion_2 positive; memory and AMP runtime probe only; full-data training still required"}
    atomic_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
