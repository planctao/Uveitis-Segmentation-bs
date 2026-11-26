import numpy as np
import pytest

from bs.severity_saliency import describe_saliency, neighbor_average, region_cues, SaliencyRankState


def test_fixed_partition_intensity_inversion_is_saliency_symmetric():
    # Rarity alone has no bright/dark direction: this is a medical interpretation limit.
    labels = np.arange(16).reshape(4, 4).repeat(4, 0).repeat(4, 1)
    gray = np.linspace(.05, .95, 16)[labels]
    first = region_cues(gray, labels)
    second = region_cues(1-gray, labels)
    np.testing.assert_allclose(first[0], second[0], rtol=1e-11, atol=1e-16)
    np.testing.assert_allclose(first[1], second[1], rtol=1e-11, atol=1e-16)


def test_region_relabeling_and_neighbor_boundary_ties_are_invariant():
    labels = np.arange(16).reshape(4, 4).repeat(4, 0).repeat(4, 1)
    gray = np.linspace(0, 1, 16)[labels]
    a, b, index = region_cues(gray, labels)
    permutation = np.random.default_rng(42).permutation(16)
    c, d, new_index = region_cues(gray, permutation[labels])
    np.testing.assert_allclose(a[index], c[new_index], atol=1e-16)
    np.testing.assert_allclose(b[index], d[new_index], atol=1e-16)
    # All candidates tie: each contributes equally, even if M is smaller than their count.
    distance = np.zeros((4, 4)); dissimilarity = np.arange(16).reshape(4, 4)
    actual = neighbor_average(dissimilarity, distance, 2)
    expected = np.array([2, 17/3, 28/3, 13])
    np.testing.assert_allclose(actual, expected)


def test_uniform_image_has_no_salient_burden_and_invalid_values_fail():
    for value in [0., .4, 1.]:
        scores, diagnostics, maps = describe_saliency(np.full((32, 32), value), True)
        assert not scores.any() and not maps.any()
    with pytest.raises(ValueError):
        describe_saliency(np.full((16, 16), np.nan))


def test_fitting_rank_handles_ties_and_keeps_fixed_thresholds():
    state = SaliencyRankState().fit([0, 0, 2, 4])
    rank, levels = state.predict([-1, 0, 1, 2, 5])
    np.testing.assert_allclose(rank, [0, .25, .5, .625, 1])
    np.testing.assert_array_equal(levels, [0, 1, 2, 2, 3])
    np.testing.assert_allclose(state.cutpoints, [0, 1, 2.5])
