"""Patient-balanced signed kernel correction in a fixed MLP representation."""
import numpy as np


def patient_support(frame, embedding, source_probability):
    z = np.asarray(embedding, dtype='float64')
    p = np.asarray(source_probability, dtype='float64')
    assert z.ndim == 2 and p.shape == (len(z), 3) and len(frame) == len(z)
    assert np.isfinite(z).all() and np.isfinite(p).all()
    assert set(frame.label) == {0, 1, 2}
    labels, patients, centers, probabilities = [], [], [], []
    grouping = frame[['patient_group', 'label']].reset_index(drop=True).groupby(['patient_group', 'label'], sort=True).indices
    for (patient, label), indices in grouping.items():
        labels.append(label);patients.append(str(patient))
        centers.append(z[indices].mean(0));probabilities.append(p[indices].mean(0))
    centers = np.stack(centers)
    labels = np.array(labels, dtype=int)
    weights = np.array([1 / (3 * np.count_nonzero(labels == label)) for label in labels])
    squared = squared_distances(centers, centers)
    positive = squared[np.triu_indices(len(squared), 1)]
    positive = positive[positive > 1e-12]
    median = max(float(np.median(positive)) if len(positive) else 1., 1e-12)
    return {'centers': centers, 'labels': labels, 'patients': np.asarray(patients),
        'source_probability': np.stack(probabilities), 'weights': weights,
        'median_squared_distance': np.array(median)}


def squared_distances(a, b):
    return np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T, 0)


def kernel_values(embedding, support, bandwidth):
    assert bandwidth > 0
    z = np.asarray(embedding, dtype='float64')
    return np.exp(-squared_distances(z, support['centers']) /
        (2 * bandwidth * float(support['median_squared_distance'])))


def fit_kernel(support, bandwidth, ridge):
    assert ridge > 0
    kernel = kernel_values(support['centers'], support, bandwidth)
    root = np.sqrt(support['weights'])
    system = root[:, None] * kernel * root[None] + ridge * np.eye(len(root))
    direct = np.eye(3)[support['labels']]
    residual = direct - support['source_probability']
    targets = np.concatenate([direct, residual], 1)
    coefficients = root[:, None] * np.linalg.solve(system, root[:, None] * targets)
    return {'direct': coefficients[:, :3], 'residual': coefficients[:, 3:],
        'bandwidth': np.array(bandwidth), 'ridge': np.array(ridge)}


def simplex_projection(values):
    values = np.asarray(values, dtype='float64')
    ordered = np.sort(values, axis=1)[:, ::-1]
    shifted = np.cumsum(ordered, 1) - 1
    count = (ordered - shifted / np.arange(1, values.shape[1] + 1) > 0).sum(1)
    threshold = shifted[np.arange(len(values)), count - 1] / count
    return np.maximum(values - threshold[:, None], 0)


def predict_kernel(support, fitted, embedding, source_probability, mode, weight):
    assert mode in ['residual', 'direct'] and 0 <= weight <= 1
    p = np.asarray(source_probability)
    if weight == 0:
        return p.copy()
    evidence = kernel_values(embedding, support, float(fitted['bandwidth'])) @ fitted[mode]
    if mode == 'residual':
        result = simplex_projection(p + weight * evidence)
    else:
        result = (1 - weight) * p + weight * simplex_projection(evidence)
    return result.astype('float32')
