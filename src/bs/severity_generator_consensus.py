"""Fixed equal-weight aggregation of already fitted synthetic directions."""
from __future__ import annotations

import copy
import numpy as np
from sklearn.preprocessing import normalize

CONSENSUS_METHODS = ('consensus_unit_direction', 'consensus_mean_rank', 'consensus_median_rank')


def empirical_rank(sorted_fit_scores, scores):
    sorted_fit_scores = np.asarray(sorted_fit_scores, dtype=np.float64)
    return (np.searchsorted(sorted_fit_scores, scores, 'left') +
            np.searchsorted(sorted_fit_scores, scores, 'right')) / (2 * len(sorted_fit_scores))


class GeneratorConsensusOrderer:
    """The generator family, weights and thresholds never use real labels."""
    def __init__(self, method):
        if method not in CONSENSUS_METHODS:
            raise ValueError(method)
        self.method = method

    def fit(self, original_fit_features, models, variants):
        if len(models) != len(variants) or len(models) < 2 or len(set(variants)) != len(variants):
            raise ValueError('Require at least two uniquely named fitted generators')
        z = np.asarray(original_fit_features, dtype=np.float64)
        self.n_fit = len(z)
        self.variants = tuple(variants)
        self.scaler = copy.deepcopy(models[0].scaler)
        for model in models:
            if model.method != 'gamma_orthogonal_direction':
                raise ValueError('Only the fixed joint-gamma family is admitted')
            if model.diagnostics['n_fit'] != self.n_fit or len(model.sorted_scores) != self.n_fit:
                raise ValueError('Fitting sample count mismatch')
            for key in ['mean_', 'scale_', 'var_', 'n_samples_seen_']:
                if not np.array_equal(getattr(model.scaler, key), getattr(self.scaler, key)):
                    raise ValueError('Generator feature transforms differ')
            if not np.isfinite(model.direction).all() or not np.isclose(np.linalg.norm(model.direction), 1, atol=1e-10):
                raise ValueError('Invalid unit direction')
        self.directions = np.stack([model.direction for model in models])
        self.generator_sorted_scores = np.stack([model.sorted_scores for model in models])
        self.generator_cutpoints = np.stack([model.cutpoints for model in models])
        mean_direction = self.directions.mean(0)
        norm = np.linalg.norm(mean_direction)
        if norm < 1e-10:
            raise ValueError('Equal-weight direction degenerates; do not retune weights')
        self.direction = mean_direction / norm
        raw = self.raw_score(z)
        self.sorted_scores = np.sort(raw)
        self.cutpoints = np.quantile(raw, [.25, .5, .75])
        self.diagnostics = dict(method=self.method, n_fit=self.n_fit,
            generator_count=len(models), equal_weights=True, masks_used=False,
            manual_grade_labels_used=False, previous_proxy_results_known=True,
            mean_unit_direction_norm=float(norm))
        return self

    def transform_features(self, z):
        return self.scaler.transform(normalize(np.asarray(z, dtype=np.float64)))

    def generator_predictions(self, z):
        features = self.transform_features(z)
        # Preserve each original vector dot product, including tie behavior at fit CDF boundaries.
        raw = np.column_stack([features @ direction for direction in self.directions])
        ranks = np.column_stack([empirical_rank(self.generator_sorted_scores[m], raw[:, m])
                                 for m in range(len(self.variants))])
        levels = np.column_stack([np.searchsorted(self.generator_cutpoints[m], raw[:, m], 'right')
                                  for m in range(len(self.variants))])
        return ranks, levels

    def raw_score(self, z):
        if self.method == 'consensus_unit_direction':
            return self.transform_features(z) @ self.direction
        ranks, _ = self.generator_predictions(z)
        return ranks.mean(1) if self.method == 'consensus_mean_rank' else np.median(ranks, axis=1)

    def predict(self, z):
        raw = self.raw_score(z)
        return empirical_rank(self.sorted_scores, raw), np.searchsorted(self.cutpoints, raw, 'right')

    def disagreement(self, z):
        ranks, levels = self.generator_predictions(z)
        minimum, maximum = levels.min(1), levels.max(1)
        return dict(generator_scores=ranks, generator_levels=levels,
                    rank_min=ranks.min(1), rank_max=ranks.max(1), rank_range=np.ptp(ranks, axis=1),
                    level_min=minimum, level_max=maximum, all_generators_agree_level=minimum == maximum,
                    common_level=np.where(minimum == maximum, minimum, -1))
