"""Describe missed GT pixels by existing component-size bins, without inference."""

import argparse
import json
from pathlib import Path


def read_directory(path):
    summary = json.loads((path / "summary.json").read_text())
    if summary["validation_samples"] != 444 or not summary["macro_matches_checkpoint"]:
        raise ValueError("Require complete replayed validation diagnostics")
    run = Path(summary["run"])
    run_summary = json.loads((run / "summary.json").read_text())
    fold = next(value for value in run_summary["folds"] if value["fold"] == summary["fold"])
    if fold["completed_epochs"] != 30:
        raise ValueError("Require full30epoch run")
    rows = {row["sample_id"]: row for row in map(json.loads, (path / "per_image.jsonl").read_text().splitlines())}
    if len(rows) != 444:
        raise ValueError("Duplicate or missing validation IDs")
    return summary, rows


def audit(candidate, control):
    candidate_summary, a = read_directory(candidate)
    control_summary, b = read_directory(control)
    if a.keys() != b.keys() or candidate_summary["image_size"] != control_summary["image_size"]:
        raise ValueError("Require same validation images and resolution")
    report = {"candidate": str(candidate.resolve()), "control": str(control.resolve()),
        "checkpoint_sha256": {"candidate": candidate_summary["checkpoint_sha256"], "control": control_summary["checkpoint_sha256"]},
        "validation_samples": 444, "image_size": candidate_summary["image_size"], "protocols": {},
        "scope": "Descriptive selected-checkpoint validation evidence. Original size bins/thresholds preserved, no inference, retuning or causal claim. Coverage>=10% is not one-to-one detection F1."}
    for protocol in ["fixed_0.5", "shared"]:
        bins = {name: {"GT_components": 0, "GT_pixels": 0,
                      "candidate": {"tp": 0, "fn": 0, "covered_at10pct": 0},
                      "control": {"tp": 0, "fn": 0, "covered_at10pct": 0}}
                for name in ["tiny", "small", "medium", "large"]}
        for sample_id in a:
            ar = a[sample_id]["protocols"][protocol]["lesion_2"]
            br = b[sample_id]["protocols"][protocol]["lesion_2"]
            if len(ar["components"]) != len(br["components"]):
                raise ValueError("GT components differ")
            for row in [ar, br]:
                tp = sum(round(c["pixels"] * c["coverage"]) for c in row["components"])
                fn = sum(c["pixels"] for c in row["components"]) - tp
                if tp != row["tp"] or fn != row["fn"]:
                    raise ValueError("Component pixel totals differ from existing replayed counts")
            for ca, cb in zip(ar["components"], br["components"], strict=True):
                if any(ca[key] != cb[key] for key in ["pixels", "size_bin", "image_area_fraction"]):
                    raise ValueError("Aligned GT component areas differ")
                bin_row = bins[ca["size_bin"]]
                bin_row["GT_components"] += 1
                bin_row["GT_pixels"] += ca["pixels"]
                for name, c in [("candidate", ca), ("control", cb)]:
                    tp = round(c["pixels"] * c["coverage"])
                    bin_row[name]["tp"] += tp
                    bin_row[name]["fn"] += c["pixels"] - tp
                    bin_row[name]["covered_at10pct"] += int(c["detected_at_10pct_coverage"])
        for row in bins.values():
            row["candidate_minus_control_FN"] = row["candidate"]["fn"] - row["control"]["fn"]
            row["candidate_minus_control_covered_components"] = row["candidate"]["covered_at10pct"] - row["control"]["covered_at10pct"]
            for mode in ["candidate", "control"]:
                row[mode]["pixel_recall_within_GT"] = row[mode]["tp"] / row["GT_pixels"] if row["GT_pixels"] else None
        report["protocols"][protocol] = {"bins": bins,
            "thresholds": {"candidate": .5 if protocol == "fixed_0.5" else candidate_summary["shared_threshold"],
                           "control": .5 if protocol == "fixed_0.5" else control_summary["shared_threshold"]},
            "total_FN_difference": sum(row["candidate_minus_control_FN"] for row in bins.values())}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Do not overwrite an existing audit")
    report = audit(args.candidate, args.control)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key == "protocols"}, indent=2))


if __name__ == "__main__":
    main()
