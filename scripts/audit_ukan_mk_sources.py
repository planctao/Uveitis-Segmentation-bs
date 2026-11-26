"""Compare pending U-KAN/MK FA objectives directly with pinned author source.

Checks valid-pixel loss/gradient equivalence, U-KAN base architecture and
effective optimizer groups/schedule. Does not evaluate FA performance or
claim original augmentation/ignore handling/benchmark protocol parity.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from torch import nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from scripts.run_paper_reproduction import atomic_json
from scripts.train_unet_multilabel import build_loss, build_model, build_optimizer, build_scheduler


def source(repo: Path, commit: str, relative: str) -> tuple[Path, ast.Module, dict]:
    path = repo / relative
    raw = path.read_bytes()
    pinned = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=repo)
    if raw != pinned:
        raise RuntimeError(f"Source differs from fixed commit: {path}")
    return path, ast.parse(raw), {"file": str(path), "commit": commit, "sha256": hashlib.sha256(raw).hexdigest()}


def extract(tree: ast.Module, name: str):
    node = next(n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    namespace = {"torch": torch, "nn": nn, "F": F}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<pinned-author-definition>", "exec"), namespace)
    return namespace[name]


def main() -> None:
    torch.set_num_threads(2)
    root = PROJECT_ROOT / "outputs/paper_reproduction_20261011"
    repos = root / "author_code"
    commits = {p["name"]: p["commit"] for p in json.loads((repos / "provenance.json").read_text())}
    evidence = []
    ukan_trees = {}
    for relative in ["Seg_UKAN/train.py", "Seg_UKAN/archs.py", "Seg_UKAN/kan.py", "Seg_UKAN/losses.py"]:
        path, tree, record = source(repos / "U-KAN", commits["U-KAN"], relative)
        ukan_trees[path.name] = tree
        evidence.append(record)
    _, mk_tree, mk_record = source(repos / "MK-UNet", commits["MK-UNet"], "train_polyp.py")
    evidence.append(mk_record)
    width_call = next(n for n in ast.walk(ukan_trees["train.py"])
                      if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                      and n.func.attr == "add_argument" and n.args
                      and isinstance(n.args[0], ast.Constant) and n.args[0].value == "--input_list")
    widths = ast.literal_eval(next(k.value for k in width_call.keywords if k.arg == "default"))
    config = yaml.safe_load((PROJECT_ROOT / "configs/paper_reproduction/ukan.yaml").read_text())
    if config["model"]["embed_dims"] != widths or widths != [128, 160, 256]:
        raise RuntimeError("Formal U-KAN must use the paper base / author training CLI width")
    # Only remove the unused wildcard utility import and inject the original
    # kan.py definitions. All network operations remain the original source.
    namespace = {"__name__": "pinned_ukan_reference"}
    exec(compile(ukan_trees["kan.py"], "<pinned-author-kan>", "exec"), namespace)
    architecture = ast.Module(body=[n for n in ukan_trees["archs.py"].body
                                   if not (isinstance(n, ast.ImportFrom) and n.module in {"utils", "kan"})], type_ignores=[])
    exec(compile(architecture, "<pinned-author-architecture>", "exec"), namespace)
    torch.manual_seed(43)
    model = build_model(config).eval()
    reference = namespace["UKAN"](num_classes=2, img_size=256, embed_dims=widths).eval()
    reference.load_state_dict(model.core.state_dict(), strict=True)
    with torch.no_grad():
        images = torch.randn(2, 3, 64, 64)
        actual, expected = model(images), reference(images)
    output_error = float((actual - expected).abs().max())
    if output_error != 0 or actual.shape != (2, 2, 64, 64):
        raise RuntimeError("Base U-KAN differs from the fixed author architecture")
    optimizer = build_optimizer(model, config)
    source_config = {"lr": config["train"]["learning_rate"], "weight_decay": config["train"]["weight_decay"],
                     "kan_lr": config["train"]["kan_learning_rate"], "kan_weight_decay": config["train"]["kan_weight_decay"]}
    group_loop = next(n for n in ast.walk(ukan_trees["train.py"])
                      if isinstance(n, ast.For) and isinstance(n.iter, ast.Call)
                      and isinstance(n.iter.func, ast.Attribute) and n.iter.func.attr == "named_parameters")
    group_namespace = {"model": reference, "config": source_config, "param_groups": []}
    exec(compile(ast.Module(body=[group_loop], type_ignores=[]), "<author-optimizer-groups>", "exec"), group_namespace)
    source_optimizer = torch.optim.Adam(group_namespace["param_groups"])
    names = {id(p): name.removeprefix("core.") for name, p in model.named_parameters()}
    source_names = {id(p): name for name, p in reference.named_parameters()}
    scheduler = build_scheduler(optimizer, config)
    source_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(source_optimizer, config["train"]["epochs"], eta_min=config["train"]["min_learning_rate"])
    schedule = []
    for epoch in range(config["train"]["epochs"] + 1):
        effective = {names[id(p)]: (g["lr"], g["weight_decay"]) for g in optimizer.param_groups for p in g["params"]}
        original = {source_names[id(p)]: (g["lr"], g["weight_decay"]) for g in source_optimizer.param_groups for p in g["params"]}
        if effective != original:
            raise RuntimeError(f"Effective optimizer groups differ at epoch {epoch}")
        if epoch in {0, 1, 15, 30}:
            schedule.append({"epoch": epoch, "learning_rates": [g["lr"] for g in optimizer.param_groups]})
        if epoch < config["train"]["epochs"]:
            optimizer.step(); source_optimizer.step()
            scheduler.step(); source_scheduler.step()
    author_ukan_loss = extract(ukan_trees["losses.py"], "BCEDiceLoss")()
    author_mk_loss = extract(mk_tree, "structure_loss")
    reports = {}
    for method, author_loss in [("ukan", author_ukan_loss), ("mkunet", author_mk_loss)]:
        method_config = yaml.safe_load((PROJECT_ROOT / f"configs/paper_reproduction/{method}.yaml").read_text())
        production_loss = build_loss(method_config)
        cases = []
        for seed in [2, 19, 47]:
            torch.manual_seed(seed)
            masks = torch.zeros(3, 40, 40, dtype=torch.long)
            masks[0, :25, :19] = 1
            masks[0, 10:14, 7:11] = 3
            masks[1, 20:23, 20:23] = 2
            target = torch.stack([masks.eq(1) | masks.eq(3), masks.eq(2) | masks.eq(3)], 1).float()
            logits = (torch.randn_like(target) * (seed % 3 + 1)).requires_grad_()
            actual_loss = production_loss(logits, masks)
            expected_loss = (torch.stack([author_loss(logits[:, c:c+1], target[:, c:c+1]) for c in range(2)]).mean()
                             if method == "ukan" else author_loss(logits, target))
            actual_gradient = torch.autograd.grad(actual_loss, logits, retain_graph=True)[0]
            expected_gradient = torch.autograd.grad(expected_loss, logits)[0]
            torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-6, atol=1e-7)
            cases.append({"seed": seed, "loss_error": float((actual_loss-expected_loss).abs().detach()),
                          "gradient_max_abs_error": float((actual_gradient-expected_gradient).abs().max()),
                          "finite_nonzero_lesion_gradients": all(bool(torch.isfinite(actual_gradient[:, c]).all()) and bool(actual_gradient[:, c].abs().sum() > 0) for c in range(2))})
        reports[method] = cases
    report = {"sources": evidence, "ukan_widths": widths, "ukan_parameters_two_outputs": sum(p.numel() for p in model.parameters()),
              "ukan_source_state_strict": True, "ukan_max_abs_output_error": output_error,
              "optimizer_parameter_tensor_counts": [len(g["params"]) for g in optimizer.param_groups],
              "all_parameters_grouped_once": len(names) == sum(len(g["params"]) for g in optimizer.param_groups),
              "effective_optimizer_groups_and_all_30_scheduler_steps_match_source": True, "schedule": schedule,
              "loss_gradient_comparisons": reports,
              "scope": "Fixed author base architecture; valid-pixel synthetic losses with absent, sparse, overlapping lesions. U-KAN adapts the original binary loss independently per lesion then averages. Does not prove original data/augmentations/ignore handling or FA performance."}
    output = root / "ukan_mk_source_objective_audit.json"
    atomic_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
