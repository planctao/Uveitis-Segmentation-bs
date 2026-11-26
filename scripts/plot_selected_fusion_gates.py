"""Inspect EDG gate maps on identical cases across the completed paired seeds."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from bs.recent_unet_methods import DifferenceGatedFusion
from scripts.fusion_evaluation_components import evaluation_components
from scripts.run_paper_reproduction import atomic_json
from scripts.train_unet_multilabel import set_seed, setup_logger
from scripts.verify_paper_experiment import file_sha256


def load_rows(directory: Path) -> dict:
    rows = [json.loads(line) for line in (directory / "per_image.jsonl").read_text().splitlines()]
    if len(rows) != 444 or len({r["sample_id"] for r in rows}) != 444:
        raise ValueError("Require complete unique clean f1 diagnostic rows")
    return {r["sample_id"]: r for r in rows}


def error_image(probabilities, mask, threshold, ignore):
    valid = mask != ignore
    target = ((mask == 2) | (mask == 3)) & valid
    predicted = (probabilities[1] >= np.float16(threshold)) & valid
    rgb = np.zeros((*mask.shape, 3))
    rgb[predicted & target] = (1, 1, 1)
    rgb[predicted & ~target] = (1, .15, .15)
    rgb[~predicted & target] = (.15, .4, 1)
    return rgb, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/edg_validation_20261011")
    args = parser.parse_args()
    root = (PROJECT_ROOT / args.root).resolve()
    root.relative_to(PROJECT_ROOT / "outputs")
    output = root / "diagnostics/selected_gate_maps"
    output.mkdir(parents=True, exist_ok=True)
    seeds = [42, 2026, 3407]
    rows = {s: {m: load_rows(root / "diagnostics" / f"paired_{m}_s{s}_f1_30_256")
                for m in ("edg", "equal")} for s in seeds}
    ids = set(rows[42]["edg"])
    if any(set(r) != ids for methods in rows.values() for r in methods.values()):
        raise ValueError("Selection requires identical validation samples")
    get = lambda s, m, i: rows[s][m][i]["protocols"]["shared"]["lesion_2"]
    negative = [i for i in ids if get(42, "edg", i)["target_pixels"] == 0]
    positive = list(ids - set(negative))
    fp_delta = lambda i: sum(get(s, "edg", i)["fp"] - get(s, "equal", i)["fp"] for s in seeds) / len(seeds)
    dice_delta = lambda i: sum(get(s, "edg", i)["dice"] - get(s, "equal", i)["dice"] for s in seeds) / len(seeds)
    selection = [(i, "largest negative-image FP increase")
                 for i in sorted(negative, key=lambda i: (-fp_delta(i), i))[:2]]
    selection += [(i, "largest positive-image Dice decrease")
                  for i in sorted(positive, key=lambda i: (dice_delta(i), i))[:2]]
    selection += [(i, "largest positive-image Dice increase")
                  for i in sorted(positive, key=lambda i: (-dice_delta(i), i))[:2]]
    if len({i for i, _ in selection}) != 6:
        raise ValueError("Require six distinct selected cases")
    wanted = {i for i, _ in selection}
    manifest = {"selection": [{"sample_id": i, "criterion": criterion,
                               "mean_edg_minus_equal_FP": fp_delta(i),
                               "mean_edg_minus_equal_positive_image_Dice": dice_delta(i) if i in positive else None}
                              for i, criterion in selection], "seeds": {},
                "scope": "extreme cases selected across three seeds on reused validation; qualitative gate observations, no vessel ground truth or causal interpretation"}
    torch.set_num_threads(2)
    for seed in seeds:
        run = root / "runs" / f"paired_edg_s{seed}_f1_30_256"
        config = json.loads((run / "resolved_config.json").read_text())
        summary = json.loads((root / "diagnostics" / run.name / "summary.json").read_text())
        checkpoint_path = run / "f1/checkpoints/best.pt"
        if file_sha256(checkpoint_path) != summary["checkpoint_sha256"] or not summary["macro_matches_checkpoint"]:
            raise ValueError("Checkpoint differs from completed diagnostics")
        set_seed(int(config["project"]["seed"]))
        logger = setup_logger(output / f"seed{seed}.log", f"selected_gate_maps_{seed}")
        build_loader, build_model, _, _ = evaluation_components(config)
        loader = build_loader(config, "f1", "val", logger)
        assert len(loader.dataset) == 444
        model = build_model(config)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint
        device = torch.device(config["runtime"]["device"])
        model = model.to(device).eval()
        fusions = [m for m in model.modules() if isinstance(m, DifferenceGatedFusion)]
        assert len(fusions) == 5
        for module in fusions:
            module.capture_diagnostics = True
        cases = {}
        with torch.inference_mode():
            for batch in loader:
                selected = wanted & set(batch["sample_id"])
                if not selected:
                    continue
                with torch.amp.autocast("cuda", enabled=bool(config["train"]["amp"]) and device.type == "cuda"):
                    probabilities = model(batch["image"].to(device)).sigmoid()
                for index, sample_id in enumerate(batch["sample_id"]):
                    if sample_id not in selected:
                        continue
                    scores = probabilities[index].float().cpu().numpy().astype(np.float16)
                    mask = batch["mask"][index].cpu().numpy()
                    valid = mask != config["data"]["ignore_index"]
                    for protocol, threshold in [("fixed_0.5", .5), ("shared", summary["shared_threshold"])]:
                        for lesion in (1, 2):
                            target = ((mask == lesion) | (mask == 3)) & valid
                            predicted = (scores[lesion - 1] >= np.float16(threshold)) & valid
                            counts = {"tp": int((predicted & target).sum()), "fp": int((predicted & ~target).sum()),
                                      "fn": int((~predicted & target).sum())}
                            recorded = rows[seed]["edg"][sample_id]["protocols"][protocol][f"lesion_{lesion}"]
                            if any(counts[k] != recorded[k] for k in counts):
                                raise ValueError(f"Selected case replay differs: {seed} {sample_id} {protocol} lesion{lesion}")
                    maps = {stage: fusions[stage].last_diagnostics["local_weight"][index, 0].float().cpu().numpy()
                            for stage in (0, 4)}
                    image = batch["image"][index].cpu()
                    image = (image * torch.tensor([.229, .224, .225])[:, None, None]
                             + torch.tensor([.485, .456, .406])[:, None, None]).clamp(0, 1).permute(1, 2, 0).numpy()
                    cases[sample_id] = {"image": image, "scores": scores, "mask": mask, "maps": maps}
                    np.savez_compressed(output / f"s{seed}_{sample_id}.npz", local_gate_deep=maps[0],
                                        local_gate_shallow=maps[4], probabilities=scores, mask=mask, display_image=image)
                for module in fusions:
                    module.last_diagnostics = {}
                if set(cases) == wanted:
                    break
        if set(cases) != wanted:
            raise ValueError("Missing selected cases")
        fig, axes = plt.subplots(6, 5, figsize=(13, 15), constrained_layout=True)
        for row_index, (sample_id, criterion) in enumerate(selection):
            case = cases[sample_id]
            with np.load(rows[seed]["equal"][sample_id]["probabilities_path"]) as cached:
                assert np.array_equal(case["mask"], cached["mask"])
                equal_threshold = json.loads((root / "diagnostics" / f"paired_equal_s{seed}_f1_30_256/summary.json").read_text())["shared_threshold"]
                equal_errors, target = error_image(cached["probabilities"], case["mask"], equal_threshold, config["data"]["ignore_index"])
            edg_errors, _ = error_image(case["scores"], case["mask"], summary["shared_threshold"], config["data"]["ignore_index"])
            axes[row_index, 0].imshow(case["image"])
            short_criterion = "FP increase" if row_index < 2 else ("Dice decrease" if row_index < 4 else "Dice increase")
            axes[row_index, 0].set_title(f"{sample_id}: {short_criterion}", fontsize=9)
            axes[row_index, 1].imshow(equal_errors)
            axes[row_index, 2].imshow(edg_errors)
            for column, stage in ((3, 4), (4, 0)):
                h, w = case["mask"].shape
                heatmap = axes[row_index, column].imshow(case["maps"][stage], vmin=0, vmax=1, cmap="viridis",
                                                        extent=(-.5, w - .5, h - .5, -.5), interpolation="nearest")
                if target.any():
                    axes[row_index, column].contour(target, levels=[.5], colors="cyan", linewidths=.5)
            for axis in axes[row_index]:
                axis.set_axis_off()
        for column, title in enumerate((f"FA input\n{selection[0][0]}: FP increase", "equal lesion_2 error", "EDG lesion_2 error", "shallow local gate", "deep local gate")):
            axes[0, column].set_title(title, fontsize=10)
        fig.colorbar(heatmap, ax=axes[:, 3:].ravel().tolist(), shrink=.5, label="channel-mean fusion weight; not confidence")
        fig.suptitle(f"EDG seed {seed}: identical cases selected by three-seed mean differences\n"
                     "White TP / red FP / blue FN; cyan lesion_2 GT; gates are fusion weights", fontsize=11)
        fig.savefig(output / f"s{seed}_gate_cases.png", dpi=140)
        fig.savefig(output / f"s{seed}_gate_cases.pdf")
        plt.close(fig)
        manifest["seeds"][str(seed)] = {"checkpoint_sha256": summary["checkpoint_sha256"],
                                       "selected_cases": len(cases), "all_selected_fixed_and_shared_counts_match": True}
        print(f"Seed {seed}: exported six gate cases; both lesion count replays passed", flush=True)
        del model, fusions, loader, cases, batch, probabilities
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    atomic_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
