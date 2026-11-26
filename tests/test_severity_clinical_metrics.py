import numpy as np
import pytest

from bs.severity_clinical_metrics import (bootstrap_correlations, correlation, interval,
                                         pair_credit, rating_status, resample_indices)


def test_patient_resampling_retains_whole_clusters_and_multiplicity():
    keys = np.array(['p1','p1','p2','p3','p3','p3'])
    draws = resample_indices(6, n_boot=30, patient_keys=keys)
    assert any(len(draw) != 6 for draw in draws)
    assert any(np.bincount(draw,minlength=6).max() > 1 for draw in draws)
    for draw in draws:
        counts = np.bincount(draw,minlength=6)
        for key in np.unique(keys): assert len(set(counts[keys==key])) == 1
        assert sum(counts[np.flatnonzero(keys==k)[0]] for k in np.unique(keys)) == 3


def test_same_draws_preserve_paired_equivalent_rank_difference():
    x = np.arange(8.)
    result = bootstrap_correlations(np.stack([x,np.exp(x)]),x,n_boot=30)
    assert np.allclose(result[0],result[1],equal_nan=True)
    assert interval(result[1]-result[0])['ci_low'] == 0


def test_constant_outcomes_are_undefined_and_counted():
    assert np.isnan(correlation([1,2,3],[1,1,1]))
    boot = bootstrap_correlations(np.array([[1,2,3]]),[1,1,1],n_boot=30)
    assert np.isnan(boot).all()
    assert interval(boot[0]) == dict(ci_low=None,ci_high=None,n_valid_bootstrap=0)


def test_exact_model_ties_half_credit_only_for_directional_doctor_choices():
    assert pair_credit('A_more',.3,.3) == .5
    assert pair_credit('B_more',.3,.3) == .5
    assert pair_credit('A_more',.8,.3) == 1
    assert pair_credit('B_more',.8,.3) == 0
    for choice in ['tie','not_comparable','ungradable','']:
        assert np.isnan(pair_credit(choice,.8,.3))


def test_swapping_both_pair_sides_leaves_credit_unchanged():
    for a,b in [(.2,.8),(.9,.1),(.4,.4)]:
        assert pair_credit('A_more',a,b) == pair_credit('B_more',b,a)


def test_missing_and_ungradable_are_distinct_and_no_grades_invented():
    rubric = dict(rubric_version='test-only',ordered_levels=['low','high'])
    row = dict(gradable='',image_inflammation_grade='',rater_id='',rubric_version='')
    assert rating_status(row,'A',rubric) == 'unscored'
    row.update(gradable='yes',rater_id='A',rubric_version='test-only')
    assert rating_status(row,'A',rubric) == 'missing_grade'
    row['gradable'] = 'no'
    assert rating_status(row,'A',rubric) == 'ungradable'
    row['image_inflammation_grade'] = 'high'
    with pytest.raises(ValueError): rating_status(row,'A',rubric)
    row['gradable'] = 'yes'
    assert rating_status(row,'A',rubric) == 'valid'
    with pytest.raises(ValueError): rating_status(row,'B',rubric)
