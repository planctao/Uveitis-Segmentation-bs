"""Prepare an independent GT-region replication and whole-image control."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from scripts.run_dino_component_transfer_gate import assert_frozen
from scripts.run_paper_reproduction import atomic_json


def read(path):
    return json.loads(path.read_text())


def screen_evidence(overnight):
    screen = overnight / "bicc_screen"
    queue, diagnostics = read(screen / "queue_state.json"), read(screen / "diagnostic_state.json")
    if queue["status"] == "failed" or diagnostics["status"] == "failed":
        raise RuntimeError("Original BiCC queue failed")
    if queue["status"] != "complete" or diagnostics["status"] != "complete":
        return "pending", {"reason": "All three original BiCC arms and replays must finish"}
    assert_frozen(screen)
    scores, reports, recipe = {}, {}, None
    for mode in ["bidirectional", "gt_only", "smoothed_pred"]:
        name = f"bicc_lesion2_{mode}_s42_f1_30_256"
        evidence_path = overnight / "completed_evidence" / f"{name}.json"
        if not evidence_path.exists():
            return "pending", {"reason": "Full additional pairing evidence pending", "job": name}
        evidence = read(evidence_path)
        pairing, replay = evidence["pairing"], evidence["replay"]
        if pairing["completed_epochs"] != 30 or pairing["original_common_tensors_matched"] != 292:
            raise RuntimeError("Incomplete original initialization/budget evidence")
        if not pairing["per_epoch_target_totals_match"] or not replay["all_required_metrics_match"] or replay["validation_samples"] != 444:
            raise RuntimeError("Original pairing or replay failed")
        summary = read(screen / "diagnostics" / name / "summary.json")
        if evidence["checkpoint_sha256"] != summary["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint mismatch")
        scores[mode] = summary["reported_macro"]["shared"]
        reports[mode] = str(evidence_path)
        if mode == "gt_only":
            recipe = read(Path(queue["jobs"][name]["run"]) / "resolved_config.json")
    scores["plain"] = read(PROJECT_ROOT / "outputs/edg_validation_20261011/diagnostics/paired_concat_s42_f1_30_256/summary.json")["reported_macro"]["shared"]
    original_bidirectional_gate = all(scores["bidirectional"] > scores[k] for k in ["plain", "gt_only", "smoothed_pred"])
    followup_gate = all(scores["gt_only"] > scores[k] for k in ["plain", "bidirectional", "smoothed_pred"])
    return "passed" if followup_gate else "not_supported", {
        "complete_seed42_scores": scores, "reports": reports, "recipe": recipe,
        "original_bidirectional_gate_passed": original_bidirectional_gate,
        "new_direction": "GT-region loss selected after original screen; original BiCC rule unchanged",
        "known_limitation": "seed42 GT-only increased pixel Dice but decreased one-to-one component F1"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    overnight = PROJECT_ROOT / "outputs/overnight_unet_20261012"
    gate, evidence = screen_evidence(overnight)
    if args.check_only:
        print(json.dumps({"gate": gate, "evidence": {k: v for k, v in evidence.items() if k != "recipe"}}, indent=2))
        return
    if gate != "passed":
        raise RuntimeError("Independent GT follow-up complete-screen gate not met")
    import torch
    import yaml
    from bs.paired_experiments import initialize_paired_model, initial_weight_fingerprints
    from scripts.train_cc_ground_truth_followup import build_model
    torch.set_num_threads(2)
    root = overnight / "cc_gt_followup"
    root.mkdir(exist_ok=True)
    if (root / "plan.json").exists() or (root / "queue_state.json").exists():
        raise RuntimeError("Do not overwrite an existing follow-up plan")
    jobs, audits = [], {}
    for seed, mode in [(42, "global_only"), (2026, "gt_only"), (3407, "gt_only")]:
        baseline = PROJECT_ROOT / f"outputs/edg_validation_20261011/runs/paired_concat_s{seed}_f1_30_256"
        original = read(baseline / "resolved_config.json")
        reference = read(baseline / "f1/initial_weight_fingerprints.json")
        config = copy.deepcopy(original)
        name = f"ccfollowup_lesion2_{mode}_s{seed}_f1_30_256"
        config["project"]["name"] = name
        config["outputs"]["root"] = str(root.relative_to(PROJECT_ROOT) / "runs")
        config["model"]["name"] = "cc_ground_truth_followup_controlled"
        weight = evidence["recipe"]["model"]["bidirectional_component_supervision"]["weight"]
        config["model"]["cc_ground_truth_followup"] = {"mode": mode, "weight": weight}
        config["reproduction"] = copy.deepcopy(evidence["recipe"]["reproduction"])
        config["reproduction"].update(
            experiment_class="independent_gt_region_followup", seed=seed, auxiliary_mode=mode,
            selection="post-screen new GT direction; not a relaxation of original BiCC gate",
            seed42_complete_scores=evidence["complete_seed42_scores"],
            control="same-weight per-image zero-smoothing Dice+BCE on lesion2, without Voronoi cells")
        for section in ["data", "train", "loss", "metric", "augmentations", "runtime"]:
            if config[section] != original[section]:
                raise RuntimeError(f"Same-seed original recipe changed: {section}")
        model = initialize_paired_model(build_model, config)
        if initial_weight_fingerprints(model) != reference:
            raise RuntimeError("Original same-seed initial tensors differ")
        audits[name] = {"original_initial_tensors_matched": len(reference),
                       "parameters": sum(p.numel() for p in model.parameters()),
                       "auxiliary_weight": weight, "original_recipe_matched": True}
        path = root / "input_configs" / f"{name}.yaml"
        path.parent.mkdir(exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        jobs.append({"id": name, "config": str(path.relative_to(PROJECT_ROOT)), "fold": "f1",
                     "training_script": "scripts/train_cc_ground_truth_followup.py",
                     "comparison_group": str(seed),
                     "reference_diagnostics": f"outputs/edg_validation_20261011/diagnostics/paired_concat_s{seed}_f1_30_256"})
        del model
    atomic_json(root / "plan.json", {
        "jobs": jobs, "created_epoch": time.time(), "complete_seed42_scores": evidence["complete_seed42_scores"],
        "original_bidirectional_gate_passed": evidence["original_bidirectional_gate_passed"],
        "advance_rule": "Complete all three new 30epoch arms even if the control wins. GT-only must exceed same-seed plain in all three seeds and exceed seed42 global-only before considering strong transfer. Review component FP/F1. Region-specific attribution remains limited to seed42: new seeds have existing plain controls, not replicated global-only controls. No independent test or stable >0.80 claim.",
        "expected_initial_difference_suffixes": [], "training_started_at_preparation": False})
    sources = list(read(overnight / "bicc_screen/source_manifest.json")["source_sha256"])
    sources += ["src/bs/cc_ground_truth_followup.py", "scripts/train_cc_ground_truth_followup.py",
                "scripts/prepare_cc_ground_truth_followup.py", "scripts/run_cc_ground_truth_followup.py",
                "scripts/probe_cc_ground_truth_followup.py", "scripts/fusion_evaluation_components.py"]
    sources += [job["config"] for job in jobs] + [str((root / "plan.json").relative_to(PROJECT_ROOT))]
    atomic_json(root / "source_manifest.json", {"created_epoch": time.time(), "source_sha256": {
        name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() for name in sources}})
    atomic_json(root / "preparation_audit.json", {"status": "prepared_not_launched", "jobs": audits,
        "scope": "Independent new direction; original BiCC source/config/gate retained unchanged"})
    print(json.dumps(audits, indent=2))


if __name__ == "__main__":
    main()
