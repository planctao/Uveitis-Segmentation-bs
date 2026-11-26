"""Aggregation properties that matter for cross-fitting and ordering."""
import copy
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler, normalize

from bs.severity_generator_consensus import CONSENSUS_METHODS, GeneratorConsensusOrderer


def fitting_problem():
    rng = np.random.default_rng(10)
    z = rng.normal(size=(31, 4))
    scaler = StandardScaler().fit(normalize(z))
    features = scaler.transform(normalize(z))
    directions = np.array([[1., 0, 0, 0], [.8, .6, 0, 0], [.8, 0, .6, 0]])
    models = []
    for direction in directions:
        raw = features @ direction
        models.append(SimpleNamespace(method='gamma_orthogonal_direction', scaler=copy.deepcopy(scaler),
            direction=direction, sorted_scores=np.sort(raw), cutpoints=np.quantile(raw, [.25, .5, .75]),
            diagnostics={'n_fit': len(z)}))
    return z, models


def test_unanimous_coordinate_order_is_preserved():
    z, models = fitting_problem()
    samples = np.array([[-1., 0, 0, 1], [0., 0, 0, 1], [1., 0, 0, 1]])
    for method in CONSENSUS_METHODS:
        model = GeneratorConsensusOrderer(method).fit(z, models, ['a', 'b', 'c'])
        ranks, levels = model.generator_predictions(samples)
        assert (np.diff(ranks, axis=0) >= 0).all()
        assert (np.diff(model.raw_score(samples)) > 0).all()
        score, level = model.predict(samples)
        assert (np.diff(score) >= 0).all() and (np.diff(level) >= 0).all()


def test_inference_cannot_change_fitting_thresholds():
    z, models = fitting_problem()
    for method in CONSENSUS_METHODS:
        model = GeneratorConsensusOrderer(method).fit(z, models, ['a', 'b', 'c'])
        before = model.cutpoints.copy()
        reference = model.predict(z[:4])
        model.predict(np.tile([1.e10, 0, 0, 1], (100, 1)))
        repeated = model.predict(z[:4])
        assert np.array_equal(before, model.cutpoints)
        assert all(np.array_equal(a, b) for a, b in zip(reference, repeated))


def test_mismatched_fit_transforms_and_cancelled_directions_are_rejected():
    z, models = fitting_problem()
    changed = copy.deepcopy(models)
    changed[1].scaler.mean_[0] += .01
    with pytest.raises(ValueError, match='transforms'):
        GeneratorConsensusOrderer(CONSENSUS_METHODS[0]).fit(z, changed, ['a', 'b', 'c'])
    opposite = copy.deepcopy(models[:2])
    opposite[1].direction = -opposite[0].direction
    with pytest.raises(ValueError, match='degenerates'):
        GeneratorConsensusOrderer(CONSENSUS_METHODS[0]).fit(z, opposite, ['a', 'b'])


def test_serialization_retains_scores_levels_and_generator_disagreement(tmp_path):
    z, models = fitting_problem()
    for method in CONSENSUS_METHODS:
        model = GeneratorConsensusOrderer(method).fit(z, models, ['a', 'b', 'c'])
        path = tmp_path / f'{method}.joblib'
        joblib.dump(model, path)
        repeat = joblib.load(path)
        assert all(np.array_equal(a, b) for a, b in zip(model.predict(z), repeat.predict(z)))
        expected, actual = model.disagreement(z), repeat.disagreement(z)
        assert all(np.array_equal(expected[key], actual[key]) for key in expected)
        assert np.array_equal(expected['common_level'] >= 0, expected['all_generators_agree_level'])
