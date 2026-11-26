"""Deployment must preserve local scales when external cohort composition changes."""
import numpy as np
import pytest

from bs.severity_ordering import ImageOnlyOrderer
from bs.severity_transfer import predict_bundle


def bundle():
    states = []
    for i in range(5):
        f = np.zeros((4,20)); f[:,0] = np.array([0,1,2,3])+i
        state = ImageOnlyOrderer('mean').fit(np.zeros((4,768)),f)
        states.append((f'f{i+1}',state))
    return {'mean_intensity':states}


def inputs(values):
    f = np.zeros((len(values),20)); f[:,0] = values
    return np.zeros((len(values),768)), f


def test_known_local_cdf_average_and_saturation():
    z,f = inputs([-10,1,20])
    folds, scores = predict_bundle(bundle(),['a','b','c'],z,f)
    # At x=1 the five fixed CDFs are [3/8,1/8,0,0,0].
    np.testing.assert_array_equal(scores.score,[0,.1,1])
    np.testing.assert_array_equal(scores.cdf_boundary_folds,[5,3,5])
    assert len(folds) == 15 and 'level' not in scores.columns


def test_new_external_patients_do_not_change_existing_scores():
    z,f = inputs([1,2])
    before = predict_bundle(bundle(),['a','b'],z,f)[1].set_index('uid')
    z,f = inputs([20,2,-10,1])
    after = predict_bundle(bundle(),['new_high','b','new_low','a'],z,f)[1].set_index('uid')
    np.testing.assert_array_equal(before.score,after.loc[before.index,'score'])


def test_missing_or_duplicate_fold_rejected():
    z,f = inputs([1])
    groups = bundle(); groups['mean_intensity'][-1] = groups['mean_intensity'][0]
    with pytest.raises(ValueError,match='five distinct folds'):
        predict_bundle(groups,['a'],z,f)


def test_duplicate_ids_rejected():
    z,f = inputs([1,2])
    with pytest.raises(ValueError,match='unique'):
        predict_bundle(bundle(),['a','a'],z,f)


def test_nonfinite_or_misaligned_features_rejected():
    z,f = inputs([1]); z[0,0] = np.nan
    with pytest.raises(ValueError,match='Nonfinite'):
        predict_bundle(bundle(),['a'],z,f)
    with pytest.raises(ValueError,match='aligned'):
        predict_bundle(bundle(),['a'],np.zeros((2,768)),f)
