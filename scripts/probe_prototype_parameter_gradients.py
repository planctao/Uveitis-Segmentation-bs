"""Training-only PDR gradient probe at representative network parameters."""

import json
import logging
from pathlib import Path
import sys

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from bs.paired_experiments import initialize_paired_model, seed_paired_epoch
from bs.prototype_regularization import prototype_discriminative_terms
from scripts.train_prototype_regularized_multilabel import build_model
from scripts import train_unet_multilabel as training
from scripts.run_paper_reproduction import atomic_json


def main():
    torch.set_num_threads(2)
    root = PROJECT_ROOT / "outputs/overnight_unet_20261012"
    config = yaml.safe_load((root / "prototype_screen/input_configs/sward_pdr_joint_unit_s42_f1_30_256.yaml").read_text())
    train_loader = training.build_loader(config, "f1", "train", logging.getLogger("parameter_probe"))
    val_loader = training.build_loader(config, "f1", "val", logging.getLogger("parameter_probe"))
    seed_paired_epoch(config, 1, train_loader, val_loader)
    batch = next(iter(train_loader))
    saved = torch.load(PROJECT_ROOT / "outputs/edg_validation_20261011/runs/paired_concat_s42_f1_30_256/f1/checkpoints/best.pt",
                       map_location="cpu", weights_only=False)
    names = ["encoder.stages.0.0.convs.0.conv.weight", "encoder.stages.5.0.convs.1.conv.weight",
             "decoder.stages.4.convs.1.conv.weight", "decoder.seg_layers.4.weight"]
    reports = []
    for snapshot in ["initial_seed42", f"trained_baseline_epoch{saved['epoch']}"]:
        model = initialize_paired_model(build_model, config).train()
        if snapshot.startswith("trained"):
            model.load_state_dict(saved["model"], strict=True)
        parameters = dict(model.named_parameters())
        selected = [parameters[name] for name in names]
        logits = model(batch["image"])
        base_loss = training.build_loss(config)(logits, batch["mask"])
        terms = prototype_discriminative_terms(model._cached_feature, batch["mask"], normalize_features=True)
        auxiliary = terms["separation"] + terms["compactness"]
        base_gradients = torch.autograd.grad(base_loss, selected, retain_graph=True)
        aux_gradients = torch.autograd.grad(auxiliary, selected, allow_unused=True)
        rows = {}
        for name, base, aux in zip(names, base_gradients, aux_gradients):
            aux = torch.zeros_like(base) if aux is None else aux
            rows[name] = {"segmentation_gradient_norm": float(base.norm()), "unit_PDR_gradient_norm": float(aux.norm()),
                          "unit_PDR_gradient_norm_ratio": float(aux.norm() / base.norm().clamp_min(1e-12)),
                          "cosine": float(torch.nn.functional.cosine_similarity(base.flatten(), aux.flatten(), dim=0)),
                          "finite": bool(torch.isfinite(base).all() and torch.isfinite(aux).all())}
        reports.append({"snapshot": snapshot, "base_loss": float(base_loss.detach()),
                        "unit_PDR_loss": float(auxiliary.detach()), "parameters": rows})
        print(json.dumps(reports[-1]), flush=True)
        model._cached_feature = None
        del model, logits, base_loss, terms, auxiliary, base_gradients, aux_gradients, parameters, selected
    atomic_json(root / "prototype_semantic_preparation/parameter_gradient_probe.json", {
        "scope": "CPU FP32 paired epoch1 first training batch; no updates, no validation inference/tuning; representative parameter gradients rather than all-parameter norm",
        "sample_ids": list(batch["sample_id"]), "folds": list(batch["fold"]), "results": reports})


if __name__ == "__main__":
    main()
