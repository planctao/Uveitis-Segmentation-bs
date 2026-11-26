"""Compare FA BiCC adaptation to the pinned author's single-depth3D loss."""

import argparse
import json
from pathlib import Path
import sys
from typing import List, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from bs.bidirectional_component_supervision import bidirectional_component_terms
from scripts.run_paper_reproduction import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    research = PROJECT_ROOT / "outputs/research/unet_overnight_20261012"
    sys.path.insert(0, str(research / "bicc_author/src"))
    original_custom_op = torch.library.custom_op

    def compatible_custom_op(*a, **kw):
        decorator = original_custom_op(*a, **kw)

        def register(function):
            if function.__name__ == "get_voronoi":
                function.__annotations__["sampling"] = Optional[List[float]]
            return decorator(function)

        return register

    torch.library.custom_op = compatible_custom_op
    from bicc_loss import VectorizedBiCCLoss
    torch.library.custom_op = original_custom_op
    cases = []
    for case in ["mixed", "GT_empty", "prediction_empty", "both_empty"]:
        logits = torch.randn(2, 7, 9, requires_grad=True)
        target = torch.zeros_like(logits, dtype=torch.bool)
        if case in {"mixed", "prediction_empty"}:
            target[0, 1:3, 1:3] = True
            target[0, 5:7, 6:9] = True
            target[1, 3:5, 4:6] = True
        if case in {"prediction_empty", "both_empty"}:
            with torch.no_grad():
                logits.fill_(-4)
        for mode in ["bidirectional", "gt_only", "smoothed_pred"]:
            alpha = 0. if mode == "gt_only" else .5
            empty = "dice" if mode == "smoothed_pred" else "detached"
            smooth = 1e-5 if mode == "smoothed_pred" else 0.
            ours = bidirectional_component_terms(logits, target, alpha=alpha,
                pred_empty_cell_loss=empty, pred_smooth=smooth)["instance"]
            original = VectorizedBiCCLoss(alpha=alpha, empty_cell_loss=empty, pred_branch_dice_smooth=smooth)
            # softmax([0,z])=sigmoid(z); original dimensional validation remains unchanged.
            reference = original(torch.stack([torch.zeros_like(logits), logits], 1).unsqueeze(2),
                                 target.to(torch.int16)[:, None, None])
            ga = torch.autograd.grad(ours, logits, retain_graph=True)[0]
            gb = torch.autograd.grad(reference, logits, retain_graph=True)[0]
            difference = float((ours - reference).abs().detach())
            gradient_difference = float((ga - gb).abs().max())
            assert difference < 1e-6 and gradient_difference < 1e-6
            cases.append({"case": case, "mode": mode, "forward_absolute_difference": difference,
                          "gradient_max_difference": gradient_difference})
    report = {"author_commit": "3cf57118746715119366587c720b10ffcc3d975d", "parity_cases": cases,
        "author_annotation_shim": "Optional[list[float]] to Optional[typing.List[float]] for torch2.5 custom-op schema",
        "scope": "Author single-depth3D loss versus independent2D sigmoid implementation; all-valid synthetic inputs, no FA accuracy or whole framework reproduction; ignore extension verified separately"}
    if args.cuda:
        logits = torch.randn(8, 256, 256, device="cuda", requires_grad=True)
        target = torch.zeros_like(logits, dtype=torch.bool)
        target[:, 20:24, 20:24] = True
        with torch.amp.autocast("cuda"):
            terms = bidirectional_component_terms(logits, target)
            terms["instance"].backward()
        assert torch.isfinite(logits.grad).all() and torch.isfinite(terms["instance"])
        report["CUDA_operator_only"] = {"finite_gradient": True, "loss_dtype": str(terms["instance"].dtype),
            "gt_components": terms["gt_components"], "pred_components": terms["pred_components"],
            "scope": "Logits and objective only, CPU Voronoi labeling with CUDA differentiable reductions; no full-model memory evidence"}
    atomic_json(PROJECT_ROOT / "outputs/overnight_unet_20261012/bicc_semantic_preparation/operator_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
