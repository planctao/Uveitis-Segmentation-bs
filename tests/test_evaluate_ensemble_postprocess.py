import pytest
import torch

from scripts.evaluate_ensemble_postprocess import (
    parse_member,
    parse_weights,
    validate_member_configs,
    weighted_ensemble_logits,
)


def _config(image_size: list[int] | None = None) -> dict:
    return {
        "train": {"image_size": image_size or [768, 768]},
        "data": {
            "root": "dataset/dataset/split_dataorigin",
            "image_dir": "img",
            "mask_dir": "mask_only_itksnap",
            "ignore_index": 255,
            "label_values": [0, 1, 2, 3],
        },
    }


def test_parse_member_splits_config_and_checkpoint() -> None:
    config, checkpoint = parse_member("configs/a.yaml:runs/x/f1/checkpoints/best.pt")

    assert config.name == "a.yaml"
    assert checkpoint.name == "best.pt"


def test_parse_weights_normalizes_and_validates_count() -> None:
    assert parse_weights(None, 2) == [0.5, 0.5]
    assert parse_weights("2,1", 2) == [2 / 3, 1 / 3]
    assert parse_weights("3,1/1,3", 2) == [[0.75, 0.25], [0.25, 0.75]]

    with pytest.raises(ValueError, match="Expected 2 weights"):
        parse_weights("1,2,3", 2)


def test_validate_member_configs_rejects_mismatched_image_size() -> None:
    with pytest.raises(ValueError, match="train.image_size"):
        validate_member_configs([_config([768, 768]), _config([640, 640])])


def test_weighted_ensemble_logits_supports_probability_and_logit_average() -> None:
    logits_a = torch.full((1, 2, 2, 2), -1.0)
    logits_b = torch.full((1, 2, 2, 2), 1.0)

    prob_average = weighted_ensemble_logits([logits_a, logits_b], [0.5, 0.5], "prob")
    logit_average = weighted_ensemble_logits([logits_a, logits_b], [0.5, 0.5], "logit")

    assert torch.allclose(prob_average, torch.zeros_like(prob_average), atol=1e-6)
    assert torch.allclose(logit_average, torch.zeros_like(logit_average), atol=1e-6)


def test_weighted_ensemble_logits_supports_per_channel_weights() -> None:
    logits_a = torch.full((1, 2, 2, 2), -2.0)
    logits_b = torch.full((1, 2, 2, 2), 2.0)

    logit_average = weighted_ensemble_logits([logits_a, logits_b], [[0.75, 0.25], [0.25, 0.75]], "logit")

    assert torch.allclose(logit_average[:, 0], torch.full((1, 2, 2), -1.0))
    assert torch.allclose(logit_average[:, 1], torch.full((1, 2, 2), 1.0))


def test_weighted_ensemble_logits_rejects_bad_channel_weight_shape() -> None:
    logits_a = torch.zeros((1, 2, 2, 2))
    logits_b = torch.ones((1, 2, 2, 2))

    with pytest.raises(ValueError, match="Per-channel weights"):
        weighted_ensemble_logits([logits_a, logits_b], [[0.5, 0.5, 0.0]], "prob")
