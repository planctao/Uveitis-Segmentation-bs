"""Fit-only normal variation orthogonal to patient-balanced diagnostic contrasts.

Diagnostic contrasts are a supervised safeguard, not proof of causal disease
directions. The safeguard preserves their coordinates before output clipping.
"""
import numpy as np
from experiment import KEYS


def image_vectors(values):
    x = np.asarray(values, dtype='float64')
    return x.mean(1) if x.ndim == 3 else x


def patient_mean(frame, x, selected):
    patients = frame.patient_group.to_numpy()
    names = sorted(set(patients[selected]))
    return names, np.stack([x[selected & (patients == name)].mean(0) for name in names])


def retained_rank(eigenvalues, maximum):
    if not len(eigenvalues):
        return 0
    return int(min(maximum, np.sum(eigenvalues > max(float(eigenvalues[0]) * 1e-6, 1e-12))))


def fit_diagnostic_geometry(frame, normalized, protection, rank=32):
    assert protection in ['coarse', 'disease']
    assert set(frame.label) == {0, 1, 2}
    normal = frame.label.eq(2).to_numpy()
    if protection == 'coarse':
        groups = [(str(label), frame.label.eq(label).to_numpy()) for label in [0, 1]]
    else:
        groups = [(name, frame.disease.eq(name).to_numpy()) for name in sorted(frame.loc[~normal, 'disease'].unique())]
        for _, selected in groups:
            assert not (selected & normal).any()
    geometry = {'protection': np.array(protection), 'diagnostic_groups': np.asarray([name for name, _ in groups], dtype=str)}
    for key in KEYS:
        x = image_vectors(normalized[key])
        names, normal_centers = patient_mean(frame, x, normal)
        center = normal_centers.mean(0)
        contrasts = np.stack([patient_mean(frame, x, selected)[1].mean(0) - center for _, selected in groups])
        _, singular, right = np.linalg.svd(contrasts, full_matrices=False)
        protected_rank = retained_rank(singular ** 2, len(contrasts))
        protected = right[:protected_rank].T
        residual = normal_centers - center
        residual -= (residual @ protected) @ protected.T
        _, singular, right = np.linalg.svd(residual, full_matrices=False)
        eigenvalues = singular ** 2 / max(len(names) - 1, 1)
        usable = retained_rank(eigenvalues, rank)
        basis = np.zeros((x.shape[-1], rank), dtype='float64')
        if usable:
            basis[:, :usable] = right[:usable].T
            signs = np.sign(basis[np.abs(basis[:, :usable]).argmax(0), np.arange(usable)])
            basis[:, :usable] *= np.where(signs == 0, 1, signs)
        spectrum = np.zeros(rank, dtype='float32')
        spectrum[:usable] = eigenvalues[:usable]
        geometry[key + '_center'] = center.astype('float32')
        geometry[key + '_basis'] = basis.astype('float32')
        geometry[key + '_eigenvalues'] = spectrum
        geometry[key + '_effective_rank'] = np.array(usable, dtype='int64')
        geometry[key + '_protected_basis'] = protected.astype('float32')
        geometry[key + '_contrasts'] = contrasts.astype('float32')
    geometry['normal_patients'] = np.asarray(names, dtype=str)
    return geometry


def independent_diagnostic_geometry(frame, normalized, geometry):
    """Rebuild patient means and use Gram eigenproblems instead of feature SVD."""
    normal = frame.label.eq(2).to_numpy()
    names = sorted(frame.loc[normal, 'patient_group'].unique())
    assert np.array_equal(names, geometry['normal_patients'])
    protection = str(geometry['protection'])
    selectors = [frame.label.eq(int(name)).to_numpy() if protection == 'coarse' else frame.disease.eq(name).to_numpy()
                 for name in geometry['diagnostic_groups']]
    diagnostics = {}
    for key in KEYS:
        x = np.asarray(normalized[key], dtype='float64')
        if x.ndim == 3:
            x = np.sum(x, axis=1) / x.shape[1]

        def means(selected):
            rows = []
            for patient in sorted(frame.loc[selected, 'patient_group'].unique()):
                ids = np.flatnonzero(selected & frame.patient_group.eq(patient).to_numpy())
                rows.append(np.sum(x[ids], axis=0) / len(ids))
            return np.stack(rows)

        normals = means(normal)
        center = np.sum(normals, axis=0) / len(normals)
        contrasts = np.stack([np.sum(means(selected), axis=0) / len(means(selected)) - center for selected in selectors])
        gram = contrasts @ contrasts.T
        eigenvalues, vectors = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues)[::-1]
        values = np.maximum(eigenvalues[order], 0)
        pr = retained_rank(values, len(contrasts))
        q = contrasts.T @ vectors[:, order[:pr]] / np.sqrt(values[:pr])[None]
        stored = geometry[key + '_protected_basis'].astype('float64')
        assert stored.shape[1] == pr
        assert np.allclose(np.linalg.svd(q.T @ stored, compute_uv=False), 1, atol=2e-6)
        # Explicit least-squares residualization provides a second projector implementation.
        residual = normals - center
        residual -= (residual @ contrasts.T) @ np.linalg.pinv(gram, rcond=1e-6) @ contrasts
        values, left = np.linalg.eigh(residual @ residual.T / max(len(normals) - 1, 1))
        order = np.argsort(values)[::-1]
        values = np.maximum(values[order], 0)
        rank = retained_rank(values, 32)
        assert rank == int(geometry[key + '_effective_rank'])
        basis = geometry[key + '_basis'].astype('float64')
        if rank:
            recovered = residual.T @ left[:, order[:rank]] / np.sqrt(values[:rank] * max(len(normals) - 1, 1))[None]
            assert np.allclose(np.linalg.svd(recovered.T @ basis[:, :rank], compute_uv=False), 1, atol=2e-6)
            assert np.allclose(basis[:, :rank].T @ basis[:, :rank], np.eye(rank), atol=1e-6)
        assert np.allclose(basis.T @ stored, 0, atol=1e-6)
        assert np.allclose(basis.T @ contrasts.T, 0, atol=1e-5)
        assert not basis[:, rank:].any() and not geometry[key + '_eigenvalues'][rank:].any()
        assert np.allclose(values[:rank], geometry[key + '_eigenvalues'][:rank], atol=2e-5, rtol=2e-6)
        assert np.allclose(center, geometry[key + '_center'], atol=1e-6)
        assert np.allclose(contrasts, geometry[key + '_contrasts'], atol=1e-6)
        diagnostics[key] = {'normal_patients': len(names), 'protected_rank': pr, 'effective_suppression_rank': rank,
                            'maximum_protected_overlap': float(np.max(np.abs(basis.T @ stored), initial=0))}
    return diagnostics
