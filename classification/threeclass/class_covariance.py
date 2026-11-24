"""Patient-balanced QDA with each covariance shrunk toward shared evidence."""
import numpy as np
from patient_gaussian import fit_evidence


def fit_class_covariance(frame, embedding, class_weight, shrinkage=.1):
    assert 0 <= class_weight <= 1
    z = np.asarray(embedding, dtype='float64')
    shared = fit_evidence(frame, z, 'coarse', shrinkage)
    covariances, cholesky, logdet = [], [], []
    # Recover the shared residual variance before the fixed diagonal regularizer.
    shared_variance = max(float(np.trace(shared['covariance']) / z.shape[1] / 1.00001), 1e-10)
    for label in range(3):
        residuals = z - shared['means'][label]
        selected = frame.label.eq(label).to_numpy()
        patients = frame.patient_group.to_numpy()
        unique = np.unique(patients[selected])
        covariance = np.zeros((z.shape[1], z.shape[1]), dtype='float64')
        for patient in unique:
            r = residuals[selected & (patients == patient)]
            covariance += r.T @ r / len(r) / len(unique)
        regularized = ((1 - class_weight) * shared['covariance'] + class_weight *
            ((1 - shrinkage) * covariance + (shrinkage + 1e-5) * shared_variance * np.eye(z.shape[1])))
        factor = np.linalg.cholesky(regularized)
        covariances.append(regularized);cholesky.append(factor)
        logdet.append(2 * np.log(np.diag(factor)).sum())
    return {**shared, 'class_covariance': np.stack(covariances),
        'class_cholesky': np.stack(cholesky), 'class_logdet': np.asarray(logdet),
        'shared_logdet': np.array(2 * np.log(np.diag(shared['cholesky'])).sum()),
        'class_weight': np.array(class_weight)}


def class_probability(model, embedding):
    z = np.asarray(embedding, dtype='float64')
    scores = []
    for label in range(3):
        whitened = np.linalg.solve(model['class_cholesky'][label], (z - model['means'][label]).T).T
        distance = np.square(whitened).sum(1)
        determinant = model['class_logdet'][label] - model['shared_logdet']
        scores.append(-.5 * (distance + determinant) / float(model['temperature']) + np.log(model['priors'][label]))
    scores = np.stack(scores, 1);scores -= scores.max(1, keepdims=True)
    probability = np.exp(scores);probability /= probability.sum(1, keepdims=True)
    return probability.astype('float32')
