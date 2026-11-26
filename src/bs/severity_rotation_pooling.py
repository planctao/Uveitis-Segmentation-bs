"""Fixed C4 means before calibration or over existing raw scores."""
import copy

import numpy as np

from bs.severity_synthetic import SyntheticDirectionOrderer


def rotation_mean(orbit):
    """Average exactly four views with a permutation-independent reduction."""
    orbit = np.asarray(orbit, dtype=np.float64)
    if orbit.ndim < 2 or orbit.shape[0] != 4 or not np.isfinite(orbit).all():
        raise ValueError('Require four finite views on the first axis')
    return np.sort(orbit, axis=0).mean(axis=0, dtype=np.float64)


class RotationFeatureMeanOrderer:
    def __init__(self, method, seed=42):
        self.calibrator = SyntheticDirectionOrderer(method, seed)

    def fit(self, original, low, high, control, gamma08, gamma12):
        self.calibrator.fit(*[rotation_mean(x) for x in [original, low, high, control, gamma08, gamma12]])
        self.diagnostics = dict(self.calibrator.diagnostics, aggregation='raw_feature_C4_mean',
                                weights=[.25]*4)
        return self

    def raw_score(self, orbit):
        return self.calibrator.raw_score(rotation_mean(orbit))

    def predict(self, orbit):
        return self.calibrator.predict(rotation_mean(orbit))


class RotationScoreMeanOrderer:
    def fit(self, original, frozen_model):
        self.base = copy.deepcopy(frozen_model)
        raw = self.raw_score(original)
        self.sorted_scores = np.sort(raw)
        self.cutpoints = np.quantile(raw, [.25, .5, .75])
        self.diagnostics = dict(method=self.base.method, seed=self.base.seed,
            n_fit=len(raw), aggregation='frozen_raw_score_C4_mean', weights=[.25]*4,
            base_direction_refitted=False, masks_used=False, manual_grade_labels_used=False)
        return self

    def raw_score(self, orbit):
        orbit = np.asarray(orbit)
        if orbit.ndim != 3 or orbit.shape[0] != 4:
            raise ValueError('Require four (sample, feature) views')
        return rotation_mean(np.stack([self.base.raw_score(view) for view in orbit]))

    def predict(self, orbit):
        raw = self.raw_score(orbit)
        score = (np.searchsorted(self.sorted_scores, raw, 'left') +
                 np.searchsorted(self.sorted_scores, raw, 'right')) / (2*len(self.sorted_scores))
        return score, np.searchsorted(self.cutpoints, raw, 'right')
