from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.verify_paper_experiment import SUMMARY_FIELDS
from scripts.verify_remaining_paper_screens import replay_matches


def reports():
    metrics = {key: 0.7 for key in SUMMARY_FIELDS}
    verified = {"run": "/runs/model", "fold": "f1", "best_epoch": 20,
                "checkpoints": {"best": {"sha256": "selected-checkpoint"}},
                "resolved_config_sha256": "config", "metric_history_sha256": "history", "metrics": metrics}
    replay = {"run": "/runs/model", "fold": "f1", "best_epoch": 20, "checkpoint_sha256": "selected-checkpoint",
              "resolved_config_sha256": "config", "metric_history_sha256": "history", "validation_samples": 444,
              "validation_augmented_excluded": True, "all_required_metrics_match": True, "reported": metrics,
              "measured": {field: 0.7 for field in SUMMARY_FIELDS.values()}}
    return verified, replay


def test_success_flag_cannot_hide_changed_actual_lesion_score():
    verified, replay = reports()
    assert replay_matches(replay, verified)
    changed = deepcopy(replay)
    changed["measured"]["paper_dice_2"] = 0.5
    assert not replay_matches(changed, verified)
    changed["measured"]["paper_dice_2"] = float("nan")
    assert not replay_matches(changed, verified)


@pytest.mark.parametrize("field", ["checkpoint_sha256", "resolved_config_sha256", "metric_history_sha256", "fold", "run", "best_epoch", "validation_samples"])
def test_matching_scores_do_not_accept_replay_from_different_inputs(field):
    verified, replay = reports()
    replay[field] = "different-input"
    assert not replay_matches(replay, verified)


def test_partial_replay_cannot_complete_full_candidate_result():
    verified, replay = reports()
    del replay["measured"]["paper_sweep_ind_dice_2"]
    assert not replay_matches(replay, verified)
    assert not replay_matches(None, verified)
