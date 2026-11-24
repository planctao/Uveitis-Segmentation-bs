"""Check geometry invariance, patient duplication and hierarchical priors."""
import numpy as np
import pandas as pd
from experiment import ROOT, dump
from patient_gaussian import fit_evidence, evidence_probability, mix_probability


def main():
    rng = np.random.default_rng(20261124)
    targets = np.repeat(np.arange(4), 6)
    parents = np.array([0, 0, 1, 2])
    frame = pd.DataFrame({'disease': ['d' + str(i) for i in targets], 'label': parents[targets],
        'patient_group': ['p' + str(i // 2) for i in range(len(targets))]})
    z = rng.normal(size=(24, 6)) + targets[:, None] * .5
    query = rng.normal(size=(9, 6))
    rotation, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    offset = rng.normal(size=(6,))
    duplicates = np.r_[np.arange(24), [0, 1]]
    for granularity in ['coarse', 'subtype']:
        for shrinkage in [.1, .5]:
            model = fit_evidence(frame, z, granularity, shrinkage)
            p = evidence_probability(model, query)
            assert np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1)
            for transformed_z, transformed_query in [(z + offset, query + offset),
                                                     (z @ rotation, query @ rotation), (7 * z, 7 * query)]:
                other = fit_evidence(frame, transformed_z, granularity, shrinkage)
                assert np.allclose(p, evidence_probability(other, transformed_query), atol=1e-6)
            duplicate_model = fit_evidence(frame.iloc[duplicates], z[duplicates], granularity, shrinkage)
            assert np.allclose(p, evidence_probability(duplicate_model, query), atol=1e-6)
            for parent in range(3):
                assert abs(model['priors'][model['parents'] == parent].sum() - 1 / 3) < 1e-12
            copied = {key: np.array(value, copy=True) for key, value in model.items()}
            assert np.array_equal(p, evidence_probability(copied, query))
            baseline = np.full_like(p, 1 / 3)
            assert np.array_equal(mix_probability(baseline, p, 0), baseline)
            assert np.array_equal(mix_probability(baseline, p, 1), p)
    dump(ROOT / 'verification_patient_gaussian.json', {'passed': True,
        'checks': ['Finite normalized probabilities for coarse and subtype evidence',
                   'Predictions invariant to translation, orthogonal coordinate changes and global feature scaling',
                   'Duplicating all images of one patient leaves evidence unchanged',
                   'Each requested parent receives exactly one-third prior mass',
                   'Model array restoration is exact', 'Mixture endpoints preserve the original or Gaussian predictions exactly']})
    print('Patient-balanced Gaussian evidence invariants passed', flush=True)


if __name__ == '__main__':
    main()
