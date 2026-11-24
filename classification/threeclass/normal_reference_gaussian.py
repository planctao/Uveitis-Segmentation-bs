"""Gaussian evidence from the normal-reference MLP embedding, mixed with M6."""
import numpy as np
import torch
from experiment import KEYS, Model, normalize
from normal_subspace import NormalSuppressedModel, independent_transform
from audit_gaussian_evidence import patient_moments


@torch.inference_mode()
def collect_source(c, features, indices, independent=False):
    assert len(indices) and max(indices) < 2091
    if independent:
        model = Model(c['cfg'], c['dims'], len(c['diseases'])).eval()
        model.load_state_dict({key: value for key, value in c['state_dict'].items() if not key.startswith('normal_')}, strict=True)
    else:
        model = NormalSuppressedModel(c['cfg'], c['dims'], len(c['diseases'])).eval()
        model.load_state_dict(c['state_dict'], strict=True)
    embeddings, probabilities = [], []
    for start in range(0, len(indices), 256):
        ids = indices[start:start + 256]
        data, _ = normalize({key: value[ids] for key, value in features.items()}, None, c['stats'])
        if independent:
            data = {key: independent_transform(data[key], c['state_dict']['normal_center_' + key].numpy(),
                    c['state_dict']['normal_basis_' + key].numpy(), c['cfg']['normal_strength']) for key in KEYS}
        out = model({key: torch.from_numpy(value) for key, value in data.items()})
        embeddings.append(out['embedding'].numpy()); probabilities.append(out['p'].numpy())
    return np.concatenate(embeddings), np.concatenate(probabilities)


def verify_evidence(frame, fit_embedding, fitted, granularity, shrinkage):
    means, covariance, parents, counts = patient_moments(frame, np.asarray(fit_embedding, dtype='float64'), granularity)
    variance = max(float(np.trace(covariance) / len(covariance)), 1e-10)
    regularized = (1 - shrinkage) * covariance + (shrinkage + 1e-5) * variance * np.eye(len(covariance))
    priors = np.array([counts[index] / counts[parents == parents[index]].sum() / 3 for index in range(len(counts))])
    assert np.allclose(fitted['means'], means, atol=1e-6, rtol=1e-6)
    assert np.allclose(fitted['covariance'], regularized, atol=1e-6, rtol=1e-6)
    assert np.allclose(fitted['cholesky'] @ fitted['cholesky'].T, regularized, atol=1e-6, rtol=1e-6)
    assert np.array_equal(fitted['parents'], parents) and np.array_equal(fitted['patient_counts'], counts)
    assert np.allclose(fitted['priors'], priors, atol=1e-12)
    assert float(fitted['temperature']) == np.sqrt(fit_embedding.shape[1])
    assert float(fitted['shrinkage']) == shrinkage
    return means, regularized, parents, priors


def independent_probability(moments, embedding):
    means, covariance, parents, priors = moments
    z = np.asarray(embedding, dtype='float64')
    # Direct covariance solves and explicit residual distances, no Cholesky score expansion.
    scores = []
    for index, mean in enumerate(means):
        residual = z - mean
        distance = np.sum(residual * np.linalg.solve(covariance, residual.T).T, axis=1)
        scores.append(-.5 * distance / np.sqrt(z.shape[1]) + np.log(priors[index]))
    scores = np.stack(scores, 1); scores -= scores.max(1, keepdims=True)
    posterior = np.exp(scores); posterior /= posterior.sum(1, keepdims=True)
    probability = np.zeros((len(z), 3), dtype='float64')
    for index, parent in enumerate(parents):
        probability[:, parent] += posterior[:, index]
    return probability.astype('float32')


def composite_probability(raw, normal, gaussian):
    return (.9 * (.25 * raw + .75 * normal) + .1 * gaussian).astype('float32')
