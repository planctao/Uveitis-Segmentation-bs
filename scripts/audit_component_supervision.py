"""Check CATMIL author binary operators and the explicit FA2D adaptation."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from bs.component_supervision import component_detection_terms
from scripts.run_paper_reproduction import atomic_json


def author_namespace(path, class_name):
    selected = {"_safe_squeeze_target", "_foreground_probability", class_name}
    definitions = [node for node in ast.parse(path.read_text()).body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in selected]
    namespace = {"np": np, "torch": torch, "nn": nn}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace[class_name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="Also check a small CUDA AMP backward without training")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    research = PROJECT_ROOT / "outputs/research/unet_overnight_20261012"
    source = research / "catmil_author/slsseg/nnunetv2/training/nnUNetTrainer"
    author_cat = author_namespace(source / "nnUNetTrainerCAT.py", "ComponentAdaptiveTverskyLoss")(w_bg=.01)
    author_mil = author_namespace(source / "nnUNetTrainerMIL.py", "SmallLesionMILLoss")()
    cases = []
    for negative_second_image in [False, True]:
        logits = torch.randn(2, 1, 4, 6, 7, requires_grad=True)
        target = torch.zeros_like(logits, dtype=torch.long)
        target[0, 0, 0, 0:2, 0:2] = 1
        target[0, 0, 3, 4:6, 5:7] = 1
        if not negative_second_image:
            target[1, 0, 1:3, 2:4, 2:4] = 1
        ours = component_detection_terms(logits.sigmoid()[:, 0], target[:, 0].bool())
        reference_cat, reference_mil = author_cat(logits, target), author_mil(logits, target)
        gradient_a = torch.autograd.grad(ours["cat"] + ours["mil"], logits, retain_graph=True)[0]
        gradient_b = torch.autograd.grad(reference_cat + reference_mil, logits)[0]
        differences = {"cat": float((ours["cat"] - reference_cat).abs().detach()),
                       "mil": float((ours["mil"] - reference_mil).abs().detach()),
                       "gradient_max_difference": float((gradient_a - gradient_b).abs().max())}
        assert max(differences.values()) < 1e-6
        cases.append({"negative_second_image": negative_second_image, "components": ours["component_count"],
                      "differences": differences})
    target2d = torch.zeros(2, 1, 8, 8, dtype=torch.long)
    target2d[:, :, 2:4, 2:4] = 1
    logits2d = torch.randn(2, 1, 8, 8, requires_grad=True)
    try:
        author_mil(logits2d, target2d)
        author_2d = {"raised": False}
    except (IndexError, RuntimeError) as error:
        author_2d = {"raised": True, "error_type": type(error).__name__, "error": str(error)}
    ours2d = component_detection_terms(logits2d.sigmoid()[:, 0], target2d[:, 0].bool())
    assert ours2d["component_count"] == 2
    report = {"author_commit": json.loads((research / "catmil_author_provenance.json").read_text())["commit"],
        "binary3D_parity_cases": cases, "author_MIL_direct2D_input": author_2d,
        "explicit2D_per_image_components": ours2d["component_count"],
        "scope": "AST-selected unchanged author loss classes; parity of binary3D operators only, not a complete training reproduction. Author is a3D implementation; direct2D failure does not invalidate its paper.",
        "source_sha256": hashlib.sha256((PROJECT_ROOT / "src/bs/component_supervision.py").read_bytes()).hexdigest()}
    if args.cuda:
        device = torch.device("cuda")
        torch.use_deterministic_algorithms(True, warn_only=True)
        logits = torch.randn(8, 1, 256, 256, device=device, requires_grad=True)
        target = torch.zeros_like(logits[:, 0], dtype=torch.bool)
        target[:4, 20:24, 20:24] = True
        with torch.amp.autocast("cuda"):
            terms = component_detection_terms(logits[:, 0].sigmoid(), target)
            loss = .1 * terms["cat"] + .2 * terms["mil"]
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
        report["small_CUDA_AMP_probe"] = {"finite": True, "loss": float(loss.detach()),
            "loss_dtype": str(loss.dtype), "components": terms["component_count"],
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "scope": "operator and logits only; not a whole-model memory or FA accuracy run"}
    output = PROJECT_ROOT / "outputs/overnight_unet_20261012/component_semantic_preparation/operator_audit.json"
    atomic_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
