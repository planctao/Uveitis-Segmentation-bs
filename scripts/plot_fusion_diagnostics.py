"""Export controlled fusion error plots and deterministically selected cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def rows(directory: Path) -> dict:
    return {row["sample_id"]: row for row in map(json.loads, (directory / "per_image.jsonl").read_text().splitlines())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/edg_validation_20261011/diagnostics")
    parser.add_argument("--methods", default="concat,fdg,edg")
    parser.add_argument("--directory-template", default="original_{method}")
    parser.add_argument("--candidate", default="edg")
    parser.add_argument("--control", default="concat")
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()
    root = Path(args.root)
    methods = args.methods.split(",")
    if args.candidate not in methods or args.control not in methods:
        parser.error("Candidate and control must be included in methods")
    if "/" in args.prefix or "\\" in args.prefix:
        parser.error("Prefix must be a file-name prefix")
    directories = {name: root / args.directory_template.format(method=name) for name in methods}
    controls = {name: json.loads((directory / "summary.json").read_text()) for name, directory in directories.items()}
    image_size = controls[args.candidate]["image_size"]
    if any(summary["image_size"] != image_size or not summary["macro_matches_checkpoint"] for summary in controls.values()):
        raise RuntimeError("Plot requires matched resolution and replayed diagnostics")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    positions = np.arange(4)
    width = 0.8 / len(methods)
    for index, lesion in enumerate(("lesion_1", "lesion_2")):
        for offset, (name, summary) in enumerate(controls.items()):
            sizes = summary["errors"]["shared"][lesion]["component_recall_by_size"]
            axes[index].bar(positions + (offset - (len(methods) - 1) / 2) * width,
                            [sizes[size]["recall_at_10pct_coverage"] for size in ("tiny", "small", "medium", "large")],
                            width=width, label=name)
        axes[index].set_xticks(positions, ("tiny", "small", "medium", "large"))
        axes[index].set_ylim(0, 1.06)
        axes[index].set_title(lesion)
        axes[index].set_ylabel("GT component recall at 10% coverage")
    axes[0].legend(loc="upper left", fontsize=8)
    for name in controls:
        curves = json.loads((directories[name] / "probability_curves.json").read_text())
        curve = curves["lesion_2"]
        axes[2].plot(curve["recall"], curve["precision"], label=f"{name} (binned AP {curve['approximate_pixel_average_precision']:.3f})")
    axes[2].set(xlabel="Pixel recall", ylabel="Pixel precision", title="lesion_2 PR curve", xlim=(0, 1), ylim=(0, 1))
    axes[2].legend(fontsize=8)
    fig.suptitle(f"f1 / {image_size[0]}: selected checkpoints; exploratory validation analysis", fontsize=11)
    fig.savefig(root / f"{args.prefix}error_comparison.png", dpi=180)
    fig.savefig(root / f"{args.prefix}error_comparison.pdf")
    plt.close(fig)
    candidate, control = rows(directories[args.candidate]), rows(directories[args.control])
    if set(candidate) != set(control):
        raise RuntimeError("Candidate and control sample IDs differ")
    eligible = [sample_id for sample_id, row in candidate.items() if row["protocols"]["shared"]["lesion_2"]["target_pixels"] > 0]
    ranked = sorted(eligible, key=lambda sample_id: candidate[sample_id]["protocols"]["shared"]["lesion_2"]["dice"]
                    - control[sample_id]["protocols"]["shared"]["lesion_2"]["dice"])
    selection = ranked[:3] + ranked[-3:][::-1]
    fig, axes = plt.subplots(6, 4, figsize=(11, 15), constrained_layout=True)
    manifest = []
    for index, sample_id in enumerate(selection):
        a, b = candidate[sample_id], control[sample_id]
        with np.load(a["probabilities_path"]) as saved:
            mask = saved["mask"]
            target = (mask == 2) | (mask == 3)
            candidate_prediction = saved["probabilities"][1] >= np.float16(controls[args.candidate]["shared_threshold"])
        with np.load(b["probabilities_path"]) as saved:
            if not np.array_equal(mask, saved["mask"]):
                raise RuntimeError("Candidate and control masks differ")
            control_prediction = saved["probabilities"][1] >= np.float16(controls[args.control]["shared_threshold"])
        height, width = mask.shape
        image = np.asarray(Image.open(a["image_path"]).convert("RGB").resize((width, height), Image.Resampling.BILINEAR)) / 255
        axes[index, 0].imshow(image)
        axes[index, 1].imshow(target, cmap="gray", vmin=0, vmax=1)
        for col, pred in ((2, control_prediction), (3, candidate_prediction)):
            # White TP, red FP, blue FN; no anatomy inferred from the colors.
            error = np.zeros((height, width, 3))
            error[pred & target] = (1, 1, 1)
            error[pred & ~target] = (1, .15, .15)
            error[~pred & target] = (.15, .4, 1)
            axes[index, col].imshow(error)
        label = f"{args.candidate} worse" if index < 3 else f"{args.candidate} better"
        axes[index, 0].set_title(f"{sample_id}: {label}", fontsize=10)
        axes[index, 1].set_title("lesion_2 GT", fontsize=10)
        axes[index, 2].set_title(f"{args.control} Dice {b['protocols']['shared']['lesion_2']['dice']:.3f}", fontsize=10)
        axes[index, 3].set_title(f"{args.candidate} Dice {a['protocols']['shared']['lesion_2']['dice']:.3f}", fontsize=10)
        for axis in axes[index]:
            axis.set_axis_off()
        manifest.append({"sample_id": sample_id, "selection": label,
                         "control": args.control, "candidate": args.candidate,
                         "control_dice": b["protocols"]["shared"]["lesion_2"]["dice"],
                         "candidate_dice": a["protocols"]["shared"]["lesion_2"]["dice"]})
    fig.suptitle("Largest lesion_2 Dice decreases/increases: white TP, red FP, blue FN", fontsize=12)
    fig.savefig(root / f"{args.prefix}lesion2_paired_cases.png", dpi=150)
    fig.savefig(root / f"{args.prefix}lesion2_paired_cases.pdf")
    (root / f"{args.prefix}lesion2_paired_cases.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
