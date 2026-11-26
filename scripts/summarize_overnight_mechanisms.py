"""Summarize complete, replayed runs without using unfinished best scores."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.audit_instance_detection import audit as instance_audit
from scripts.run_paper_reproduction import atomic_json


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    root = PROJECT_ROOT / "outputs/overnight_unet_20261012"
    output = root / "complete_run_comparison"
    output.mkdir(exist_ok=True)
    rows, pending = [], []
    reference_cache = {}
    cached_instances = read(root / "instance_detection_reference_set.json")["jobs"]
    for path in root.glob("*/*instance_detection.json"):
        instance = read(path)
        if "diagnostics" in instance:
            cached_instances[Path(instance["diagnostics"]).name] = instance

    def cached_instance(diagnostic, checkpoint_sha256):
        value = cached_instances.get(diagnostic.name)
        if value and value["checkpoint_sha256"] == checkpoint_sha256:
            return value
        value = instance_audit(diagnostic)
        cached_instances[diagnostic.name] = value
        atomic_json(output / f"instance_{diagnostic.name}.json", value)
        return value

    for path in output.glob("instance_*.json"):
        instance = read(path)
        cached_instances[Path(instance["diagnostics"]).name] = instance
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not (directory / "plan.json").exists():
            continue
        plan = read(directory / "plan.json")
        queue = read(directory / "queue_state.json") if (directory / "queue_state.json").exists() else {"jobs": {}}
        for specification in plan.get("jobs", []):
            name = specification["id"]
            job = queue["jobs"].get(name, {})
            diagnostic = directory / "diagnostics" / name
            replay_path = directory / "verified_experiments" / f"{name}_replay.json"
            if job.get("status") != "complete" or not replay_path.exists() or not (diagnostic / "summary.json").exists():
                pending.append({"queue": directory.name, "job": name, "status": job.get("status", "not_started"),
                                "summary_score": None, "reason": "Only full training and full replay are admitted"})
                continue
            replay = read(replay_path)
            if replay["validation_samples"] != 444 or not replay["all_required_metrics_match"]:
                raise RuntimeError(f"Incomplete/inconsistent replay: {name}")
            summary = read(diagnostic / "summary.json")
            if summary["checkpoint_sha256"] != replay["checkpoint_sha256"]:
                raise RuntimeError("Diagnostic checkpoint mismatch")
            run = Path(job["run"])
            config = read(run / "resolved_config.json")
            with (run / job["fold"] / "metrics.csv").open() as stream:
                history = list(csv.DictReader(stream))
            if len(history) != 30:
                raise RuntimeError("Comparison requires all30epochs")
            seed = int(re.search(r"_s(\d+)_", name).group(1))
            resolution = int(config["train"]["image_size"][0])
            is_baseline = directory.name == "strong_baseline_512"
            reference_name = specification.get("reference_diagnostics")
            if reference_name is None and not is_baseline and resolution == 256:
                reference_name = f"outputs/edg_validation_20261011/diagnostics/paired_concat_s{seed}_f1_30_256"
            reference = (PROJECT_ROOT / reference_name).resolve() if reference_name else None
            # Prefer durable completed-evidence reports. Recompute descriptive
            # component counts only for earlier seed42 screens without them.
            evidence_path = root / "completed_evidence" / f"{name}.json"
            evidence = read(evidence_path) if evidence_path.exists() else None
            instance = evidence.get("instance_detection") if evidence else None
            if instance is None:
                instance = cached_instance(diagnostic, summary["checkpoint_sha256"])
            shared_instance = instance["protocols"]["shared"]["pooled_all444"]
            row = {"queue": directory.name, "job": name, "seed": seed, "resolution": resolution,
                "completed_epochs": 30, "selected_epoch": job["metrics"]["best_epoch"],
                "fixed_macro": summary["reported_macro"]["fixed_0.5"], "shared_macro": summary["reported_macro"]["shared"],
                "instance_f1": shared_instance["f1_all_images"], "instance_FP_components": shared_instance["false_positive_components"],
                "instance_matched_GT": shared_instance["matched"], "instance_GT_components": shared_instance["gt_components"],
                "shared_macro_delta_vs_plain": None, "fixed_macro_delta_vs_plain": None,
                "instance_f1_delta_vs_plain": None, "reference_diagnostics": str(reference) if reference else None,
                "replay_max_absolute_difference": max(replay["absolute_differences"].values()),
                "latest_optimizer_updates": None, "latest_skipped_optimizer_updates": None}
            if evidence and evidence.get("optimizer"):
                latest = evidence["optimizer"]["checkpoints"]["latest"]
                row.update(latest_optimizer_updates=latest["executed_encoder_updates"],
                           latest_skipped_optimizer_updates=latest["skipped_encoder_updates"])
            if reference:
                key = str(reference)
                if key not in reference_cache:
                    control_summary = read(reference / "summary.json")
                    reference_cache[key] = {"summary": control_summary,
                        "instance": cached_instance(reference, control_summary["checkpoint_sha256"])}
                control = reference_cache[key]
                row.update(shared_macro_delta_vs_plain=row["shared_macro"] - control["summary"]["reported_macro"]["shared"],
                    fixed_macro_delta_vs_plain=row["fixed_macro"] - control["summary"]["reported_macro"]["fixed_0.5"],
                    instance_f1_delta_vs_plain=row["instance_f1"] - control["instance"]["protocols"]["shared"]["pooled_all444"]["f1_all_images"])
            rows.append(row)
    report = {"snapshot_utc": datetime.now(timezone.utc).isoformat(), "complete_replayed_runs": rows,
        "unfinished_or_unreplayed_runs": pending,
        "scope": "Descriptive paired deltas on the same444-image model-selection fold; no independent test/significance claim. Input resolutions and same-seed baselines are kept distinct. Partial training scores excluded; original gates unchanged."}
    atomic_json(output / "summary.json", report)
    if rows:
        with (output / "complete_runs.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        candidates = [row for row in rows if row["resolution"] == 256 and row["shared_macro_delta_vs_plain"] is not None]
        if candidates:
            labels = [re.sub(r"_s\d+_f1_30_256$", "", row["job"]).replace("mcads_", "DSUB: ").replace("acm_mswt_", "Wavelet: ").replace("catmil_lesion2_", "CATMIL: ").replace("bicc_lesion2_", "BiCC: ") + f" (s{row['seed']})" for row in candidates]
            fig, axes = plt.subplots(1, 2, figsize=(12, max(4, len(candidates) * .35)), sharey=True)
            for axis, field, title in zip(axes, ["shared_macro_delta_vs_plain", "instance_f1_delta_vs_plain"],
                                         ["Shared-threshold Macro Dice", "One-to-one lesion2 component F1"]):
                values = [100 * row[field] for row in candidates]
                axis.barh(range(len(candidates)), values, color=["#2d8a66" if value > 0 else "#bb5555" for value in values], height=.7)
                axis.axvline(0, color="#555555", linewidth=.8)
                axis.set_title(title)
                axis.set_xlabel("Difference from same-seed plain (percentage points)")
                axis.grid(axis="x", alpha=.15)
                axis.set_axisbelow(True)
            axes[0].set_yticks(range(len(candidates)), labels)
            axes[0].invert_yaxis()
            fig.suptitle("Complete 30-epoch runs at 256, verified by checkpoint replay\nModel-selection fold; component matching: IoU > 0.1, 2D 8-connectivity", fontsize=12)
            fig.tight_layout()
            fig.savefig(output / "paired_dice_component_deltas.png", dpi=160)
            fig.savefig(output / "paired_dice_component_deltas.pdf")
            plt.close(fig)
    print(json.dumps({"complete_replayed_runs": len(rows), "pending_runs": len(pending), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
