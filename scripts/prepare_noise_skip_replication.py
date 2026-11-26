"""Prepare matched noise-location seeds only after complete positive screening."""

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
    screen = overnight / "noise_skip_screen"
    if not (screen / "queue_state.json").exists():
        return "pending", {"reason": "No NoiseUNet accuracy queue started"}
    queue = read(screen / "queue_state.json")
    diagnostic_path = screen / "diagnostic_state.json"
    if queue["status"] == "failed":
        raise RuntimeError("Noise screening training failed")
    if not diagnostic_path.exists():
        return "pending", {"reason": "Screen diagnostics pending"}
    diagnostic = read(diagnostic_path)
    if diagnostic["status"] == "failed":
        raise RuntimeError("Noise screening diagnostics failed")
    if queue["status"] != "complete" or diagnostic["status"] != "complete":
        return "pending", {"reason": "Require both full30epoch location arms and replays"}
    assert_frozen(screen)
    names = {mode: f"noiseunet_{mode}_uniform_d0p5_s42_f1_30_256" for mode in ["skip", "encoder"]}
    if set(queue["jobs"]) != set(names.values()):
        raise RuntimeError("Unexpected noise screen arms")
    scores, reports, recipes = {}, {}, {}
    for mode, name in names.items():
        evidence_path = overnight / "completed_evidence" / f"{name}.json"
        if not evidence_path.exists():
            return "pending", {"reason": "Extra completed pairing evidence pending", "job": name}
        report = read(evidence_path)
        if report["pairing"]["completed_epochs"] != 30 or report["pairing"]["original_common_tensors_matched"] != 292:
            raise RuntimeError("Full-budget original initialization evidence invalid")
        if not report["pairing"]["per_epoch_target_totals_match"] or not report["replay"]["all_required_metrics_match"] or report["replay"]["validation_samples"] != 444:
            raise RuntimeError("Screen pairing/replay evidence invalid")
        summary = read(screen / "diagnostics" / name / "summary.json")
        if summary["checkpoint_sha256"] != report["checkpoint_sha256"]:
            raise RuntimeError("Screen checkpoint and evidence differ")
        scores[mode] = summary["reported_macro"]["shared"]
        reports[mode] = str(evidence_path)
        recipes[mode] = read(Path(queue["jobs"][name]["run"]) / "resolved_config.json")
    plain = PROJECT_ROOT / "outputs/edg_validation_20261011/diagnostics/paired_concat_s42_f1_30_256/summary.json"
    scores["plain"] = read(plain)["reported_macro"]["shared"]
    passed = scores["skip"] > scores["plain"] and scores["skip"] > scores["encoder"]
    return "passed" if passed else "not_supported", {"complete_seed42_scores": scores, "reports": reports, "recipes": recipes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    overnight = PROJECT_ROOT / "outputs/overnight_unet_20261012"
    gate, evidence = screen_evidence(overnight)
    if args.check_only:
        print(json.dumps({"gate": gate, "evidence": {k: v for k, v in evidence.items() if k != "recipes"}}, indent=2))
        return
    if gate != "passed":
        raise RuntimeError(f"Original complete noise screen gate not met: {gate}")
    import torch
    import yaml
    from bs.paired_experiments import initialize_paired_model, initial_weight_fingerprints
    from scripts.train_noise_skip_multilabel import build_model
    torch.set_num_threads(2)
    root = overnight / "noise_skip_replication"
    root.mkdir(exist_ok=True)
    if (root / "plan.json").exists() or (root / "queue_state.json").exists():
        raise RuntimeError("Cannot overwrite existing replication plan")
    jobs, audits = [], {}
    for seed in [2026, 3407]:
        baseline = PROJECT_ROOT / f"outputs/edg_validation_20261011/runs/paired_concat_s{seed}_f1_30_256"
        original = read(baseline / "resolved_config.json")
        reference = read(baseline / "f1/initial_weight_fingerprints.json")
        for mode in ["skip", "encoder"]:
            config = copy.deepcopy(original)
            name = f"noiseunet_{mode}_uniform_d0p5_s{seed}_f1_30_256"
            config["project"]["name"] = name
            config["outputs"]["root"] = str(root.relative_to(PROJECT_ROOT) / "runs")
            config["model"]["noise_skip"] = copy.deepcopy(evidence["recipes"][mode]["model"]["noise_skip"])
            config["reproduction"] = copy.deepcopy(evidence["recipes"][mode]["reproduction"])
            config["reproduction"].update(experiment_class="paired_bounded_noise_location_replication", seed=seed,
                                          complete_seed42_scores=evidence["complete_seed42_scores"])
            for section in ["data", "train", "loss", "metric", "augmentations", "runtime"]:
                if config[section] != original[section]:
                    raise RuntimeError(f"Same-seed recipe changed: {section}")
            model = initialize_paired_model(build_model, config)
            if initial_weight_fingerprints(model) != reference:
                raise RuntimeError("Original same-seed initial tensors differ")
            audits[name] = {"original_initial_tensors_matched": len(reference),
                           "parameters": sum(p.numel() for p in model.parameters()),
                           "bound_and_location_unchanged": True}
            path = root / "input_configs" / f"{name}.yaml"
            path.parent.mkdir(exist_ok=True)
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            jobs.append({"id": name, "config": str(path.relative_to(PROJECT_ROOT)), "fold": "f1",
                         "training_script": "scripts/train_noise_skip_multilabel.py", "comparison_group": str(seed),
                         "reference_diagnostics": f"outputs/edg_validation_20261011/diagnostics/paired_concat_s{seed}_f1_30_256"})
            del model
    atomic_json(root / "plan.json", {"jobs": jobs, "created_epoch": time.time(),
        "complete_seed42_scores": evidence["complete_seed42_scores"], "expected_initial_difference_suffixes": [],
        "advance_rule": "Skip must exceed corresponding plain and encoder control in all three complete matched seeds before formal strong transfer.",
        "training_started_at_preparation": False})
    sources = list(read(overnight / "noise_skip_screen/source_manifest.json")["source_sha256"])
    sources += ["scripts/prepare_noise_skip_replication.py", "scripts/run_noise_skip_replication_gate.py"]
    sources += [job["config"] for job in jobs] + [str((root / "plan.json").relative_to(PROJECT_ROOT))]
    atomic_json(root / "source_manifest.json", {"created_epoch": time.time(), "source_sha256": {
        name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() for name in sources}})
    atomic_json(root / "preparation_audit.json", {"status": "prepared_not_launched", "jobs": audits,
        "complete_seed42_scores": evidence["complete_seed42_scores"], "scope": "Actual initialization/config preparation, no accuracy result"})
    print(json.dumps(audits, indent=2))


if __name__ == "__main__":
    main()
