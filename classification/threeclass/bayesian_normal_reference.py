"""Seed-dependent Bayesian-bootstrap normal-patient geometry for M5 heads."""
import numpy as np
from experiment import KEYS
from normal_subspace import patient_centers
from scale_normal_geometry import supported_basis


def patient_weights(count, scope, concentration, phase_index, seed):
    assert scope in ['shared_stream_weights', 'independent_stream_weights'] and concentration > 0
    rng = np.random.default_rng(20281924 + 100 * phase_index + seed)
    if scope == 'shared_stream_weights':
        shared = rng.dirichlet(np.full(count, concentration))
        return {key: shared.copy() for key in KEYS}
    return {key: rng.dirichlet(np.full(count, concentration)) for key in KEYS}


def fit_bayesian_geometry(frame, normalized, scope, concentration, phase_index, seed, weights_override=None):
    names, _ = patient_centers(frame, normalized, 'global')
    assert len(names) > 1
    weights = patient_weights(len(names), scope, concentration, phase_index, seed) if weights_override is None else weights_override
    result = {'normal_patients': np.asarray(names, dtype=str), 'scope': np.array(scope),
              'concentration': np.array(concentration), 'phase_index': np.array(phase_index), 'model_seed': np.array(seed)}
    for key in KEYS:
        patients, centers = patient_centers(frame, normalized, key); assert patients == names
        w = np.asarray(weights[key], dtype='float64')
        assert w.shape == (len(names),) and (w > 0).all() and abs(w.sum() - 1) < 1e-12
        weighted_center = np.sum(centers * w[:, None], axis=0)
        correction = 1 - np.square(w).sum(); assert correction > 1e-8
        rows = (centers - weighted_center) * np.sqrt(w / correction)[:, None]
        basis, eigenvalues, rank = supported_basis(rows, 32)
        result[key + '_center'] = centers.mean(0).astype('float32')
        result[key + '_basis'] = basis
        result[key + '_eigenvalues'] = eigenvalues
        result[key + '_effective_rank'] = np.array(rank, dtype='int64')
        result[key + '_weights'] = w
        result[key + '_weighted_center'] = weighted_center
    return result


def independent_bayesian_geometry(frame, normalized, geometry, check_draws=True):
    normal = frame.label.eq(2).to_numpy(); names = sorted(frame.loc[normal, 'patient_group'].unique())
    assert np.array_equal(names, geometry['normal_patients'])
    if check_draws:
        rng = np.random.default_rng(20281924 + 100 * int(geometry['phase_index']) + int(geometry['model_seed']))
        if str(geometry['scope']) == 'shared_stream_weights':
            draws = rng.dirichlet(np.full(len(names), float(geometry['concentration'])))
            for key in KEYS: assert np.array_equal(draws, geometry[key + '_weights'])
        else:
            for key in KEYS:
                assert np.array_equal(rng.dirichlet(np.full(len(names), float(geometry['concentration']))), geometry[key + '_weights'])
            assert not np.array_equal(geometry['global_weights'], geometry['regions_weights'])
    diagnostics = {}
    for key in KEYS:
        x = np.asarray(normalized[key], dtype='float64')
        if x.ndim == 3: x = np.sum(x, axis=1) / x.shape[1]
        centers = []
        for patient in names:
            ids = np.flatnonzero(normal & frame.patient_group.eq(patient).to_numpy())
            centers.append(np.sum(x[ids], axis=0) / len(ids))
        centers = np.stack(centers); w = geometry[key + '_weights']
        mean = np.zeros(centers.shape[1], dtype='float64')
        for index in range(len(w)): mean += w[index] * centers[index]
        correction = 1 - np.dot(w, w)
        a = np.stack([(centers[index] - mean) * (w[index] / correction) ** .5 for index in range(len(w))])
        values, left = np.linalg.eigh(a @ a.T); order = np.argsort(values)[::-1]; values = np.maximum(values[order], 0)
        rank = int(min(32, np.sum(values > max(float(values[0]) * 1e-6, 1e-12))))
        assert rank == int(geometry[key + '_effective_rank'])
        basis = geometry[key + '_basis'].astype('float64')
        if rank:
            recovered = a.T @ left[:, order[:rank]] / np.sqrt(values[:rank])[None]
            cosine = np.linalg.svd(recovered.T @ basis[:, :rank], compute_uv=False)
            assert np.allclose(cosine, 1, atol=2e-6)
            assert np.allclose(basis[:, :rank].T @ basis[:, :rank], np.eye(rank), atol=1e-6)
            error = float(np.max(np.abs(cosine - 1)))
        else: error = 0.
        assert not basis[:, rank:].any()
        assert np.allclose(values[:rank], geometry[key + '_eigenvalues'][:rank], atol=2e-5, rtol=2e-6)
        assert np.allclose(centers.mean(0), geometry[key + '_center'], atol=1e-6)
        assert np.allclose(mean, geometry[key + '_weighted_center'], atol=1e-10)
        diagnostics[key] = {'patients': len(names), 'effective_rank': rank, 'maximum_principal_cosine_error': error,
                            'effective_weighted_patients': float(1 / np.square(w).sum())}
    return diagnostics
