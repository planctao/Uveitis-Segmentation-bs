"""Fit-normal patient PCA with separate or tied axes across two DINO scales."""
import numpy as np
import torch
from experiment import KEYS, Model
from normal_subspace import patient_centers


def supported_basis(rows, rank):
    _, singular, right = np.linalg.svd(rows, full_matrices=False)
    values = singular ** 2
    usable = int(min(rank, np.sum(values > max(float(values[0]) * 1e-6, 1e-12))))
    basis = np.zeros((rows.shape[1], rank), dtype='float64')
    if usable:
        basis[:, :usable] = right[:usable].T
        signs = np.sign(basis[np.abs(basis[:, :usable]).argmax(0), np.arange(usable)])
        basis[:, :usable] *= np.where(signs == 0, 1, signs)
    spectrum = np.zeros(rank, dtype='float32'); spectrum[:usable] = values[:usable]
    return basis.astype('float32'), spectrum, usable


def fit_scale_normal_geometry(frame, normalized, mode, rank=32):
    assert mode in ['separate', 'tied']
    result = {'scale_mode': np.array(mode)}
    for key in KEYS:
        assert normalized[key].shape[-1] == 1536
        names, centers = patient_centers(frame, normalized, key)
        centers = centers.reshape(len(names), 2, 768)
        center = centers.mean(0)
        residual = centers - center
        if mode == 'tied':
            rows = residual.reshape(-1, 768) / np.sqrt(2 * (len(names) - 1))
            basis, eigenvalues, usable = supported_basis(rows, rank)
            bases = np.stack([basis, basis]); spectra = np.stack([eigenvalues, eigenvalues])
            ranks = [usable, usable]
        else:
            parts = [supported_basis(residual[:, scale] / np.sqrt(len(names) - 1), rank) for scale in range(2)]
            bases = np.stack([part[0] for part in parts]); spectra = np.stack([part[1] for part in parts])
            ranks = [part[2] for part in parts]
        result[key + '_center'] = center.astype('float32')
        result[key + '_basis'] = bases
        result[key + '_eigenvalues'] = spectra
        result[key + '_effective_rank'] = np.array(ranks, dtype='int64')
    result['normal_patients'] = np.asarray(names, dtype=str)
    return result


def independent_scale_geometry(frame, normalized, geometry):
    normal = frame.label.eq(2).to_numpy(); names = sorted(frame.loc[normal, 'patient_group'].unique())
    assert np.array_equal(names, geometry['normal_patients'])
    mode = str(geometry['scale_mode']); diagnostics = {}
    for key in KEYS:
        x = np.asarray(normalized[key], dtype='float64')
        if x.ndim == 3:
            x = np.sum(x, axis=1) / x.shape[1]
        centers = []
        for name in names:
            ids = np.flatnonzero(normal & frame.patient_group.eq(name).to_numpy())
            centers.append(np.sum(x[ids], axis=0) / len(ids))
        means = np.stack(centers).reshape(len(names), 2, 768)
        center = np.sum(means, axis=0) / len(names)
        assert np.allclose(center, geometry[key + '_center'], atol=1e-6)
        residual = means - center
        errors = []
        for scale in range(2):
            if mode == 'tied':
                rows = np.concatenate([residual[:, 0], residual[:, 1]], axis=0) / np.sqrt(2 * (len(names) - 1))
            else:
                rows = residual[:, scale] / np.sqrt(len(names) - 1)
            values, left = np.linalg.eigh(rows @ rows.T)
            order = np.argsort(values)[::-1]; values = np.maximum(values[order], 0)
            rank = int(min(32, np.sum(values > max(float(values[0]) * 1e-6, 1e-12))))
            assert rank == int(geometry[key + '_effective_rank'][scale])
            basis = geometry[key + '_basis'][scale].astype('float64')
            if rank:
                recovered = rows.T @ left[:, order[:rank]] / np.sqrt(values[:rank])[None]
                cosines = np.linalg.svd(recovered.T @ basis[:, :rank], compute_uv=False)
                assert np.allclose(cosines, 1, atol=2e-6)
                assert np.allclose(basis[:, :rank].T @ basis[:, :rank], np.eye(rank), atol=1e-6)
                error = float(np.max(np.abs(cosines - 1)))
            else:
                error = 0.
            assert not basis[:, rank:].any()
            assert np.allclose(values[:rank], geometry[key + '_eigenvalues'][scale, :rank], atol=2e-5, rtol=2e-6)
            errors.append({'scale': scale, 'effective_rank': rank, 'maximum_principal_cosine_error': error})
        if mode == 'tied':
            assert np.array_equal(geometry[key + '_basis'][0], geometry[key + '_basis'][1])
        diagnostics[key] = errors
    return diagnostics


def independent_scale_transform(values, center, basis, strength):
    if strength == 0:
        return np.array(values, copy=True)
    x = np.asarray(values, dtype='float64'); shape = x.shape
    scales = x.reshape(*shape[:-1], 2, 768); output = np.empty_like(scales)
    for scale in range(2):
        b = np.asarray(basis[scale], dtype='float64')
        z = scales[..., scale, :]
        output[..., scale, :] = z - strength * ((z - center[scale]) @ b) @ b.T
    return np.clip(output.reshape(shape), -10, 10).astype('float32')


class ScaleNormalModel(Model):
    def __init__(self, cfg, dims, n_diseases, geometry=None):
        super().__init__(cfg, dims, n_diseases)
        rank = cfg['normal_scale_rank']
        assert rank in [16, 32] and cfg['normal_strength'] in [0., 1.]
        assert cfg['normal_scale_mode'] in ['separate', 'tied']
        for key in KEYS:
            assert dims[key] == 1536
            self.register_buffer('normal_center_' + key, torch.zeros(2, 768))
            self.register_buffer('normal_basis_' + key, torch.zeros(2, 768, rank))
            if geometry is not None:
                getattr(self, 'normal_center_' + key).copy_(torch.as_tensor(geometry[key + '_center']))
                getattr(self, 'normal_basis_' + key).copy_(torch.as_tensor(geometry[key + '_basis'][:, :, :rank]))

    def transform(self, x):
        if self.cfg['normal_strength'] == 0:
            return x
        result = {}
        for key, value in x.items():
            basis = getattr(self, 'normal_basis_' + key)
            scales = value.reshape(*value.shape[:-1], 2, 768)
            coordinates = torch.einsum('...sd,sdr->...sr', scales - getattr(self, 'normal_center_' + key), basis)
            correction = torch.einsum('...sr,sdr->...sd', coordinates, basis)
            result[key] = (scales - correction).reshape_as(value).clamp(-10, 10)
        return result

    def forward(self, x):
        return super().forward(self.transform(x))
