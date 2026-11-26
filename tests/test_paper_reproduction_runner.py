import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from scripts.run_paper_reproduction import completed_metrics, process_alive
from scripts.train_unet_multilabel import make_root_run_dir


def fake_run(tmp_path: Path, losses=(1., .5)) -> Path:
    fold = tmp_path / "f1"
    (fold / "checkpoints").mkdir(parents=True)
    (fold / "checkpoints" / "latest.pt").touch()
    (tmp_path / "summary.json").write_text(json.dumps({"folds": [{"fold": "f1", "best_paper_macro_dice": .8, "best_val_loss": .5}]}))
    with (fold / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows({"epoch": i, "train_loss": loss, "val_loss": loss} for i, loss in enumerate(losses, 1))
    return tmp_path


def test_terminal_summary_does_not_hide_missing_epochs_or_nonfinite_loss(tmp_path):
    run = fake_run(tmp_path)
    assert completed_metrics(run, "f1", 2) is not None
    assert completed_metrics(run, "f1", 3) is None
    assert completed_metrics(run, "f2", 2) is None
    p = run / "f1" / "metrics.csv"
    p.write_text(p.read_text().replace("0.5", "nan"))
    assert completed_metrics(run, "f1", 2) is None


def test_reused_pid_is_not_considered_original_training_process():
    with patch("scripts.run_paper_reproduction.process_start", return_value="100"):
        assert process_alive(123, "100")
        assert not process_alive(123, "99")
        assert not process_alive(123, None)


def test_same_directory_resume_keeps_existing_artifacts(tmp_path):
    run = make_root_run_dir(tmp_path, "example")
    marker = run / "marker.txt"
    marker.write_text("keep")
    assert make_root_run_dir(tmp_path, "example", resume=True) == run
    assert marker.read_text() == "keep"


def test_promotion_condseg_dependencies_never_cross_validation_folds():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from promote_paper_reproduction import make_jobs
    jobs = make_jobs("condseg", ["f1", "f2", "f3", "f4", "f5"], 768)
    assert len(jobs) == 10
    assert len({j["id"] for j in jobs}) == 10
    indexed = {j["id"]: j for j in jobs}
    for job in jobs:
        if "stage1_job" in job:
            stage1 = indexed[job["stage1_job"]]
            assert stage1["fold"] == job["fold"]
            assert stage1["overrides"]["train"]["image_size"] == [768, 768]


def external_state(path, job=None, **changes):
    state = {"jobs": {"candidate": job} if job else {}, "status": "running",
             "supervisor_pid": 123, "supervisor_start": "100"}
    state.update(changes)
    path.write_text(json.dumps(state))
    return state


def test_external_owned_pending_job_waits_until_worker_completes(tmp_path):
    from scripts.run_paper_reproduction import wait_for_external_job
    path = tmp_path / "external.json"
    external_state(path)
    completed = {"status": "complete", "return_code": 0}
    def finish(_):
        external_state(path, completed)
    with patch("scripts.run_paper_reproduction.process_alive", return_value=True), \
         patch("scripts.run_paper_reproduction.time.sleep", side_effect=finish) as sleep:
        assert wait_for_external_job("candidate", path) == completed
    sleep.assert_called_once()


@pytest.mark.parametrize("failed_job,failed_worker", [(True, False), (False, True)])
def test_external_failure_never_requests_duplicate_training(tmp_path, failed_job, failed_worker):
    from scripts.run_paper_reproduction import wait_for_external_job
    path = tmp_path / "external.json"
    external_state(path, {"status": "failed"} if failed_job else None,
                   status="failed" if failed_worker else "running")
    with pytest.raises(RuntimeError, match="External worker failed"):
        wait_for_external_job("candidate", path)


def test_dead_external_supervisor_can_be_observed_while_owned_child_lives(tmp_path):
    from scripts.run_paper_reproduction import wait_for_external_job
    path = tmp_path / "external.json"
    external_state(path, {"status": "running", "pid": 456, "process_start": "200"})
    completed = {"status": "complete", "return_code": 0}
    with patch("scripts.run_paper_reproduction.process_alive", side_effect=[False, True]), \
         patch("scripts.run_paper_reproduction.time.sleep", side_effect=lambda _: external_state(path, completed)):
        assert wait_for_external_job("candidate", path) == completed


def test_dead_external_worker_fails_without_launching_a_copy(tmp_path):
    from scripts.run_paper_reproduction import wait_for_external_job
    path = tmp_path / "external.json"
    external_state(path)
    with patch("scripts.run_paper_reproduction.process_alive", return_value=False), \
         pytest.raises(RuntimeError, match="worker and requested child are missing"):
        wait_for_external_job("candidate", path)


@pytest.mark.parametrize("return_code,valid_history,accepted", [(0, True, True), (1, True, False), (0, False, False)])
def test_primary_requires_external_success_and_full_metrics_without_launching(tmp_path, return_code, valid_history, accepted):
    from scripts.run_paper_reproduction import run_plan
    config = {"train": {"epochs": 2}, "outputs": {"root": "runs"}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    job = {"id": "candidate", "config": "config.yaml", "fold": "f1"}
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"jobs": [job]}))
    external = tmp_path / "external.json"
    external_state(external, {"status": "complete", "return_code": return_code})
    run = fake_run(tmp_path / "runs/candidate", losses=(1., .5) if valid_history else (1.,))
    state = tmp_path / "state.json"
    with patch("scripts.run_paper_reproduction.PROJECT_ROOT", tmp_path), \
         patch("scripts.run_paper_reproduction.os.chdir"), \
         patch("scripts.run_paper_reproduction.subprocess.Popen") as launch:
        if accepted:
            run_plan(plan, state, external_plan=plan, external_state=external)
            observed = json.loads(state.read_text())
            assert observed["status"] == "complete"
            assert observed["jobs"]["candidate"]["external_state"] == str(external)
            assert observed["jobs"]["candidate"]["metrics"] == completed_metrics(run, "f1", 2)
        else:
            with pytest.raises(RuntimeError, match="incomplete or invalid"):
                run_plan(plan, state, external_plan=plan, external_state=external)
        launch.assert_not_called()


def test_external_job_configuration_mismatch_is_rejected(tmp_path):
    from scripts.run_paper_reproduction import run_plan
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"train": {"epochs": 2}, "outputs": {"root": "runs"}}))
    job = {"id": "candidate", "config": "config.yaml", "fold": "f1"}
    plan, other = tmp_path / "plan.json", tmp_path / "other.json"
    plan.write_text(json.dumps({"jobs": [job]}))
    other.write_text(json.dumps({"jobs": [{**job, "fold": "f2"}]}))
    with patch("scripts.run_paper_reproduction.PROJECT_ROOT", tmp_path), \
         patch("scripts.run_paper_reproduction.os.chdir"), \
         patch("scripts.run_paper_reproduction.subprocess.Popen") as launch, \
         pytest.raises(RuntimeError, match="specification differs"):
        run_plan(plan, tmp_path / "state.json", external_plan=other, external_state=tmp_path / "external.json")
    launch.assert_not_called()


@pytest.mark.parametrize("precomputed", [False, True])
def test_promotion_delegates_controls_to_their_existing_owner(tmp_path, precomputed):
    from scripts.promote_paper_reproduction import run_matched_plan
    plan = tmp_path / "matched_resolution_plan.json"
    if precomputed:
        (tmp_path / "control_precompute_plan.json").write_text("{}")
        (tmp_path / "control_precompute_state.json").write_text("{}")
    with patch("scripts.promote_paper_reproduction.run_plan") as run:
        run_matched_plan(tmp_path, plan, True)
    run.assert_called_once_with(plan, tmp_path / "matched_resolution_state.json", True,
                               external_plan=tmp_path / "control_precompute_plan.json" if precomputed else None,
                               external_state=tmp_path / "control_precompute_state.json" if precomputed else None)


@pytest.mark.parametrize("missing", ["plan", "state"])
def test_promotion_rejects_incomplete_control_ownership_without_launching(tmp_path, missing):
    from scripts.promote_paper_reproduction import run_matched_plan
    present = "state" if missing == "plan" else "plan"
    (tmp_path / f"control_precompute_{present}.json").write_text("{}")
    with patch("scripts.promote_paper_reproduction.run_plan") as run, \
         pytest.raises(RuntimeError, match="both its plan and state"):
        run_matched_plan(tmp_path, tmp_path / "matched.json", True)
    run.assert_not_called()


def test_controls_wait_for_the_live_condseg_child_and_verify_its_result(tmp_path):
    from scripts.precompute_paper_controls import wait_for_full_condseg
    state_path = tmp_path / "screen_state.json"
    running = {"status": "running", "pid": 456, "process_start": "200"}
    complete = {"status": "complete", "return_code": 0, "run": str(tmp_path / "run"), "metrics": {"score": .7}}
    state = {"status": "running", "supervisor_pid": 123, "supervisor_start": "100",
             "jobs": {"author_condseg_f1_30_256": running}}
    state_path.write_text(json.dumps(state))
    def finish(_):
        state["jobs"]["author_condseg_f1_30_256"] = complete
        state_path.write_text(json.dumps(state))
    report = {"metrics": complete["metrics"]}
    with patch("scripts.precompute_paper_controls.process_alive", side_effect=[False, True]), \
         patch("scripts.precompute_paper_controls.time.sleep", side_effect=finish) as sleep, \
         patch("scripts.precompute_paper_controls.verify_run", return_value=report) as verify:
        assert wait_for_full_condseg(tmp_path) == report
    sleep.assert_called_once()
    verify.assert_called_once_with(tmp_path / "run", "f1", 30, require_full_data=True)


def test_controls_do_not_start_after_a_failed_condseg_run(tmp_path):
    from scripts.precompute_paper_controls import wait_for_full_condseg
    (tmp_path / "screen_state.json").write_text(json.dumps({"status": "failed",
        "jobs": {"author_condseg_f1_30_256": {"status": "failed"}}}))
    with patch("scripts.precompute_paper_controls.verify_run") as verify, \
         pytest.raises(RuntimeError, match="Screen failed"):
        wait_for_full_condseg(tmp_path)
    verify.assert_not_called()
