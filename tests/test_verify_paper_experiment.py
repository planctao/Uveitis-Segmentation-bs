import csv
import json

import pytest
import torch

from scripts.verify_paper_experiment import SUMMARY_FIELDS, verify_run


def make_experiment(path):
    config = {"train": {"epochs": 2, "folds_to_run": ["f1"], "max_train_samples": 8},
              "data": {"exclude_val_augmented": True}, "model": {"name": "test"},
              "loss": {"name": "test"}, "metric": {"selection_metric": "paper_sweep_best_macro_dice"},
              "reproduction": {"experiment_class": "smoke"}}
    (path / "f1/checkpoints").mkdir(parents=True)
    (path / "resolved_config.json").write_text(json.dumps(config))
    metrics = {"loss": .3, "paper_dice_1": .6, "paper_dice_2": .8, "paper_macro_dice": .7,
               "paper_sweep_best_dice_1": .7, "paper_sweep_best_dice_2": .9, "paper_sweep_best_macro_dice": .8,
               "paper_sweep_ind_dice_1": .8, "paper_sweep_ind_dice_2": .9, "paper_sweep_ind_macro_dice": .85}
    with (path / "f1/metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss"] + [f"val_{k}" for k in metrics])
        writer.writeheader()
        writer.writerows({"epoch": epoch, "train_loss": 1. / epoch,
                          **{f"val_{k}": v / (3 - epoch) for k, v in metrics.items()}} for epoch in [1, 2])
    summary = {"fold": "f1", "completed_epochs": 2, "best_epoch": 2,
               "selection_metric": "paper_sweep_best_macro_dice",
               **{key: metrics[field] for key, field in SUMMARY_FIELDS.items()}}
    (path / "summary.json").write_text(json.dumps({"folds": [summary]}))
    for label in ["best", "latest"]:
        torch.save({"epoch": 2, "best_epoch": 2, "best_score": .8, "best_metrics": metrics,
                    "config": config, "model": {"weight": torch.tensor([1., 2.])}},
                   path / f"f1/checkpoints/{label}.pt")
    return path


def test_verified_report_covers_both_checkpoint_files_and_same_epoch_metrics(tmp_path):
    report = verify_run(make_experiment(tmp_path), "f1")
    assert report["best_epoch"] == report["epochs"] == 2
    assert set(report["checkpoints"]) == {"best", "latest"}
    assert all(row["finite_model_state"] and len(row["sha256"]) == 64 for row in report["checkpoints"].values())


@pytest.mark.parametrize("corruption,match", [
    ("epoch", "checkpoint epoch"), ("config", "loss config differs"),
    ("nonfinite", "Non-finite or invalid"), ("score", "checkpoint best score"),
    ("selected_epoch", "selected epoch"), ("metrics", "checkpoint selected paper_dice_1"),
])
def test_checkpoint_corruption_is_rejected(tmp_path, corruption, match):
    run = make_experiment(tmp_path)
    path = run / "f1/checkpoints/best.pt"
    payload = torch.load(path, weights_only=False)
    if corruption == "epoch":
        payload["epoch"] = 1
    elif corruption == "config":
        payload["config"]["loss"]["name"] = "another objective"
    elif corruption == "nonfinite":
        payload["model"]["weight"][0] = float("nan")
    elif corruption == "score":
        payload["best_score"] = .9
    elif corruption == "selected_epoch":
        payload["best_epoch"] = 1
    elif corruption == "metrics":
        payload["best_metrics"]["paper_dice_1"] = .2
    torch.save(payload, path)
    with pytest.raises(ValueError, match=match):
        verify_run(run, "f1")


def test_summary_cannot_mix_dice_columns_from_different_epochs(tmp_path):
    run = make_experiment(tmp_path)
    path = run / "summary.json"
    summary = json.loads(path.read_text())
    summary["folds"][0]["best_paper_dice_1"] = .3
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="summary best_paper_dice_1"):
        verify_run(run, "f1")


@pytest.mark.parametrize("missing", ["best", "latest"])
def test_both_checkpoint_files_are_required(tmp_path, missing):
    run = make_experiment(tmp_path)
    (run / f"f1/checkpoints/{missing}.pt").unlink()
    with pytest.raises(ValueError, match="Missing"):
        verify_run(run, "f1")


def test_missing_epoch_history_is_rejected_even_with_complete_checkpoints(tmp_path):
    run = make_experiment(tmp_path)
    path = run / "f1/metrics.csv"
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0], lines[2]]) + "\n")
    with pytest.raises(ValueError, match="Missing complete finite metric history"):
        verify_run(run, "f1")


def test_smoke_run_cannot_pass_full_data_config_gate(tmp_path):
    run = make_experiment(tmp_path)
    with pytest.raises(ValueError, match="Limited-data"):
        verify_run(run, "f1", require_full_data=True)


def test_requested_epochs_cannot_silently_shrink_the_experiment(tmp_path):
    run = make_experiment(tmp_path)
    with pytest.raises(ValueError, match="Requested epochs"):
        verify_run(run, "f1", epochs=1)


def test_promotion_rejects_incomplete_queue_without_auditing(tmp_path, monkeypatch):
    from scripts import promote_paper_reproduction as promotion
    monkeypatch.setattr(promotion, "verify_run", lambda *args, **kwargs: pytest.fail("Incomplete run audited"))
    with pytest.raises(RuntimeError, match="incomplete or empty"):
        promotion.audit_completed_queue(tmp_path, {"status": "running", "jobs": {}}, "screen")


def test_promotion_audits_actual_checkpoint_metrics_before_accepting_queue_score(tmp_path):
    from scripts.promote_paper_reproduction import audit_completed_queue
    run = make_experiment(tmp_path / "run")
    config_path = run / "resolved_config.json"
    config = json.loads(config_path.read_text())
    config["train"]["max_train_samples"] = None
    config_path.write_text(json.dumps(config))
    for label in ["best", "latest"]:
        checkpoint = run / f"f1/checkpoints/{label}.pt"
        payload = torch.load(checkpoint, weights_only=False)
        payload["config"] = config
        torch.save(payload, checkpoint)
    metrics = json.loads((run / "summary.json").read_text())["folds"][0]
    job = {"status": "complete", "run": str(run), "fold": "f1", "epochs": 2, "metrics": metrics}
    state = {"status": "complete", "jobs": {"candidate": job}}
    audit_completed_queue(tmp_path, state, "screen")
    report = json.loads((tmp_path / "verified_experiments/candidate.json").read_text())
    assert report["full_data_config_verified"]
    assert report["metrics"] == metrics
    job["metrics"] = {**metrics, "best_paper_sweep_macro_dice": .95}
    with pytest.raises(RuntimeError, match="queue score differs"):
        audit_completed_queue(tmp_path, state, "screen")


def test_fivefold_statistics_include_each_lesion_under_all_threshold_policies():
    from scripts.promote_paper_reproduction import summarize_fivefold_metrics

    fields = {f"best_{policy}_{measure}"
              for policy in ("paper", "paper_sweep", "paper_sweep_ind")
              for measure in ("dice_1", "dice_2", "macro_dice")}
    rows = [{"fold": f"f{i}", **{field: i / 10 for field in fields}} for i in range(1, 6)]
    report = summarize_fivefold_metrics(rows)
    assert set(report) == fields
    for values in report.values():
        assert values["mean"] == pytest.approx(.3)
        assert values["sd"] == pytest.approx(.025 ** .5)


@pytest.mark.parametrize("folds", [["f1", "f2", "f3", "f4"], ["f1", "f2", "f3", "f4", "f4"]])
def test_fivefold_statistics_reject_missing_or_duplicate_folds(folds):
    from scripts.promote_paper_reproduction import summarize_fivefold_metrics

    with pytest.raises(ValueError, match="one verified row"):
        summarize_fivefold_metrics([{"fold": fold} for fold in folds])
