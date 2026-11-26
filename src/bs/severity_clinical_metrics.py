"""Clinical endpoint arithmetic; no model fitting or artificial clinician labels."""
import numpy as np
from scipy.stats import spearmanr


def correlation(score, grade):
    score, grade = np.asarray(score), np.asarray(grade)
    if len(score) < 3 or np.ptp(score) == 0 or np.ptp(grade) == 0:
        return np.nan
    return float(spearmanr(score, grade).statistic)


def resample_indices(n, n_boot=2000, seed=20261013, patient_keys=None):
    """Sample complete patient clusters, retaining repeat clusters with multiplicity."""
    if n < 1:
        raise ValueError('Cannot bootstrap an empty cohort')
    rng = np.random.default_rng(seed)
    if patient_keys is None:
        return [v for v in rng.integers(n, size=(n_boot, n))]
    keys = np.asarray(patient_keys, dtype=str)
    if len(keys) != n or np.any(keys == ''):
        raise ValueError('Patient bootstrap requires every assessment patient key')
    unique = np.unique(keys)
    groups = [np.flatnonzero(keys == k) for k in unique]
    return [np.concatenate([groups[k] for k in draw])
            for draw in rng.integers(len(groups), size=(n_boot, len(groups)))]


def bootstrap_correlations(scores, grades, patient_keys=None, n_boot=2000):
    scores, grades = np.asarray(scores), np.asarray(grades)
    if scores.ndim != 2 or scores.shape[1] != len(grades):
        raise ValueError('Scores must be methods by assessment images')
    draws = resample_indices(len(grades), n_boot=n_boot, patient_keys=patient_keys)
    return np.asarray([[correlation(s[idx], grades[idx]) for idx in draws] for s in scores])


def interval(values):
    valid = np.asarray(values)[np.isfinite(values)]
    if len(valid) < 20:
        return dict(ci_low=None, ci_high=None, n_valid_bootstrap=len(valid))
    lo, hi = np.quantile(valid, [.025, .975])
    return dict(ci_low=float(lo), ci_high=float(hi), n_valid_bootstrap=len(valid))


def pair_credit(choice, score_a, score_b):
    """Doctor ties/ungradable/unscored are separate endpoints, not half-credit cases."""
    if choice not in ['A_more', 'B_more']:
        return np.nan
    delta = score_a - score_b
    if delta == 0:
        return .5
    return float((delta > 0) == (choice == 'A_more'))


def rating_status(row, rater, rubric):
    """Keep missing, explicitly ungradable, and valid ordinal observations distinct."""
    gradable, grade = row['gradable'], row['image_inflammation_grade']
    if gradable not in ['', 'yes', 'no']:
        raise ValueError('gradable must be blank, yes, or no')
    if gradable == '' and grade == '':
        return 'unscored'
    if row['rater_id'] != rater or row['rubric_version'] != rubric['rubric_version']:
        raise ValueError('Every assessed image must identify its rater and agreed rubric')
    if gradable == 'no':
        if grade != '':
            raise ValueError('An ungradable image cannot carry an inflammation grade')
        return 'ungradable'
    if gradable == 'yes' and grade == '':
        return 'missing_grade'
    if gradable != 'yes' or grade not in rubric['ordered_levels']:
        raise ValueError('A grade requires gradable=yes and an agreed ordinal level')
    return 'valid'
