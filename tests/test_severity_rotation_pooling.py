import itertools

import joblib
import numpy as np
import pytest

from bs.severity_rotation_pooling import (
    RotationFeatureMeanOrderer, RotationScoreMeanOrderer, rotation_mean,
)
from bs.severity_synthetic import SyntheticDirectionOrderer


def test_noninvariant_encoder_becomes_c4_invariant_without_reflection_claim():
    image = np.random.default_rng(12).normal(size=(7, 7))
    def encode(a):
        return np.array([a[0, 0], a[0, 0]*a[0, 1], a[2, 1]])
    def orbit(a):
        return np.stack([encode(np.rot90(a, k)) for k in range(4)])
    assert not np.array_equal(encode(image), encode(np.rot90(image)))
    pooled = rotation_mean(orbit(image))
    for k in range(4):
        assert np.array_equal(rotation_mean(orbit(np.rot90(image, k))), pooled)
    assert not np.allclose(rotation_mean(orbit(np.fliplr(image))), pooled)


def test_mean_is_equal_weight_per_image_and_bitwise_permutation_invariant():
    orbit = np.random.default_rng(2).normal(size=(4, 6, 8))
    orbit[:, 0, 0] = [1e16, -1e16, 1., 2.]
    original = orbit.copy()
    expected = rotation_mean(orbit)
    for permutation in itertools.permutations(range(4)):
        assert np.array_equal(rotation_mean(orbit[list(permutation)]), expected)
    assert np.array_equal(rotation_mean(orbit[:, :3]), expected[:3])
    assert np.array_equal(orbit, original)
    simple = np.arange(4*3*2).reshape(4, 3, 2)
    assert np.array_equal(rotation_mean(simple), simple.mean(axis=0))


def test_calibrators_fit_only_given_samples_and_replay_after_serialization(tmp_path):
    rng = np.random.default_rng(4)
    z = rng.normal(size=(4, 70, 16))
    d = np.zeros(16);d[0] = 1
    nuisance = np.zeros(16);nuisance[1] = 1
    feature = RotationFeatureMeanOrderer('synthetic_direction').fit(
        z, z+.1*d, z+.2*d, z+.2*nuisance, z+.1*nuisance, z-.1*nuisance)
    base = SyntheticDirectionOrderer('synthetic_direction').fit(
        z[0], z[0]+.1*d, z[0]+.2*d, z[0]+.2*nuisance, z[0]+.1*nuisance, z[0]-.1*nuisance)
    direction = base.direction.copy()
    score_model = RotationScoreMeanOrderer().fit(z, base)
    heldout = rng.normal(size=(4, 11, 16))
    for name, model in [('feature', feature), ('score', score_model)]:
        before = model.predict(heldout)
        fitted = model.diagnostics['n_fit']
        assert fitted == 70 and len(model.calibrator.sorted_scores if name == 'feature' else model.sorted_scores) == 70
        for k in range(4):
            after = model.predict(np.roll(heldout, k, axis=0))
            assert np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1])
        path = tmp_path/f'{name}.joblib';joblib.dump(model, path)
        reloaded = joblib.load(path).predict(heldout)
        assert np.array_equal(before[0], reloaded[0]) and np.array_equal(before[1], reloaded[1])
        assert model.diagnostics['n_fit'] == fitted
    assert np.array_equal(base.direction, direction)
    assert np.array_equal(score_model.base.direction, direction)
    score_model.base.direction[0] += 1
    assert np.array_equal(base.direction, direction)


def test_rejects_incomplete_or_nonfinite_orbit():
    for bad in [np.zeros((3, 7, 4)), np.zeros(4), np.full((4, 2, 3), np.nan)]:
        with pytest.raises(ValueError):
            rotation_mean(bad)
