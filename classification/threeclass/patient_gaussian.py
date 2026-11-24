"""Patient-balanced, tempered shared-covariance class/subtype evidence."""
import numpy as np
from experiment import sample_weights


def fit_evidence(frame, embedding, granularity, shrinkage):
    assert granularity in ['coarse', 'subtype'] and 0 < shrinkage <= 1
    z = np.asarray(embedding, dtype='float64')
    assert len(frame) == len(z) and np.isfinite(z).all() and set(frame.label) == {0, 1, 2}
    names = sorted(frame.disease.unique()) if granularity == 'subtype' else [0, 1, 2]
    targets = frame.disease.map({name: index for index, name in enumerate(names)}).to_numpy() if granularity == 'subtype' else frame.label.to_numpy()
    if granularity == 'subtype':
        mapping = frame.groupby('disease').label.agg(['first', 'nunique'])
        assert mapping['nunique'].eq(1).all()
        parents = np.array([mapping.loc[name, 'first'] for name in names], dtype=int)
    else:
        parents = np.arange(3)
    weights = sample_weights(frame, targets).astype('float64')
    means = np.stack([np.average(z[targets == index], axis=0, weights=weights[targets == index])
                      for index in range(len(names))])
    residual = z - means[targets]
    covariance = (residual * weights[:, None]).T @ residual / weights.sum()
    variance = max(float(np.trace(covariance) / z.shape[1]), 1e-10)
    regularized = (1 - shrinkage) * covariance + (shrinkage + 1e-5) * variance * np.eye(z.shape[1])
    cholesky = np.linalg.cholesky(regularized)
    counts = frame.assign(_target=targets).groupby('_target').patient_group.nunique().reindex(range(len(names))).to_numpy()
    priors = np.array([counts[index] / counts[parents == parents[index]].sum() / 3
                       for index in range(len(names))])
    assert np.isclose(priors.sum(), 1)
    return {'means': means, 'cholesky': cholesky, 'covariance': regularized, 'priors': priors,
            'parents': parents, 'names': np.asarray(names, dtype=str),
            'patient_counts': counts, 'temperature': np.array(np.sqrt(z.shape[1])),
            'shrinkage': np.array(shrinkage)}


def evidence_probability(model, embedding):
    z = np.asarray(embedding, dtype='float64')
    transformed = np.linalg.solve(model['cholesky'], z.T).T
    means = np.linalg.solve(model['cholesky'], model['means'].T).T
    distances = np.maximum((transformed ** 2).sum(1)[:, None] + (means ** 2).sum(1)[None]
                           - 2 * transformed @ means.T, 0)
    scores = -.5 * distances / float(model['temperature']) + np.log(model['priors'])[None]
    scores -= scores.max(1, keepdims=True)
    posterior = np.exp(scores)
    posterior /= posterior.sum(1, keepdims=True)
    mapping = np.eye(3)[model['parents']]
    return (posterior @ mapping).astype('float32')


def mix_probability(original, gaussian, weight):
    assert 0 <= weight <= 1 and original.shape == gaussian.shape
    return ((1 - weight) * original + weight * gaussian).astype('float32')
