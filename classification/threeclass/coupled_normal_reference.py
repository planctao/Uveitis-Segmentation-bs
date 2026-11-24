"""Fit-only shared normal factors, with image-level correction preserving regional contrasts."""
import numpy as np
import torch
from experiment import KEYS, Model
from normal_subspace import patient_centers
from scale_normal_geometry import supported_basis


def fit_coupled_geometry(frame, normalized, mode):
    assert mode in ['independent', 'coupled']
    result = {'mode': np.array(mode)}; centers = []
    for key in KEYS:
        names, c = patient_centers(frame, normalized, key)
        result[key + '_center'] = c.mean(0).astype('float32')
        centers.append(c - c.mean(0))
    assert len(names) > 1
    scales = np.array([max(float(np.sqrt(np.square(c).sum() / ((len(names) - 1) * c.shape[1]))), 1e-3) for c in centers])
    result['stream_scales'] = scales.astype('float32')
    if mode == 'coupled':
        rows = np.concatenate([c / scales[i] for i, c in enumerate(centers)], 1) / np.sqrt(len(names) - 1)
        b, v, r = supported_basis(rows, 32)
        result.update(joint_basis=b, joint_eigenvalues=v, joint_effective_rank=np.array(r))
    else:
        for key, c in zip(KEYS, centers):
            b, v, r = supported_basis(c / np.sqrt(len(names) - 1), 32)
            result[key + '_basis'] = b; result[key + '_eigenvalues'] = v; result[key + '_effective_rank'] = np.array(r)
    result['normal_patients'] = np.array(names, dtype=str)
    return result


def independent_coupled_geometry(frame, normalized, geometry):
    normal = frame.label.eq(2).to_numpy(); names = sorted(frame.loc[normal, 'patient_group'].unique())
    assert np.array_equal(names, geometry['normal_patients'])
    centers = []
    for key in KEYS:
        x = normalized[key].astype('float64')
        if x.ndim == 3: x = np.sum(x, 1) / x.shape[1]
        c = np.stack([np.sum(x[normal & frame.patient_group.eq(name).to_numpy()], 0) /
                      np.sum(normal & frame.patient_group.eq(name).to_numpy()) for name in names])
        mean = np.sum(c, 0) / len(names)
        assert np.allclose(mean, geometry[key + '_center'], atol=1e-6)
        centers.append(c - mean)
    scales = np.array([max(float(np.sqrt(np.mean(np.sum(c * c, 0) / (len(names) - 1)))), 1e-3) for c in centers])
    assert np.allclose(scales, geometry['stream_scales'], atol=1e-7, rtol=1e-7)
    if str(geometry['mode']) == 'coupled':
        rows = np.hstack([c / scales[i] for i, c in enumerate(centers)]) / np.sqrt(len(names) - 1)
        items = [('joint', rows)]
    else: items = [(key, c / np.sqrt(len(names) - 1)) for key, c in zip(KEYS, centers)]
    diagnostics = {}
    for key, rows in items:
        values, left = np.linalg.eigh(rows @ rows.T); order = np.argsort(values)[::-1]; values = np.maximum(values[order], 0)
        rank = int(min(32, np.sum(values > max(values[0] * 1e-6, 1e-12))))
        basis = geometry[key + '_basis'].astype('float64')
        assert rank == int(geometry[key + '_effective_rank']) and not basis[:, rank:].any()
        recovered = rows.T @ left[:, order[:rank]] / np.sqrt(values[:rank])[None]
        cosines = np.linalg.svd(recovered.T @ basis[:, :rank], compute_uv=False)
        assert np.allclose(cosines, 1, atol=2e-6)
        assert np.allclose(basis[:, :rank].T @ basis[:, :rank], np.eye(rank), atol=1e-6)
        assert np.allclose(values[:rank], geometry[key + '_eigenvalues'][:rank], atol=2e-5, rtol=2e-6)
        diagnostics[key] = {'patients': len(names), 'effective_rank': rank,
                            'maximum_principal_cosine_error': float(np.max(np.abs(cosines - 1))) if rank else 0.}
    return diagnostics


def independent_coupled_transform(data, geometry, mode, rank, strength=1., clip=True):
    means = {k: np.mean(v.astype('float64'), 1) if v.ndim == 3 else v.astype('float64') for k, v in data.items()}
    if mode == 'coupled':
        coordinates = np.zeros((len(means['global']), rank), dtype='float64'); offset = 0
        parts = {}
        for i, key in enumerate(KEYS):
            dim = means[key].shape[1]; parts[key] = geometry['joint_basis'][offset:offset + dim, :rank].astype('float64')
            coordinates += np.einsum('nd,dr->nr', (means[key] - geometry[key + '_center']) / geometry['stream_scales'][i], parts[key])
            offset += dim
        corrections = {k: np.einsum('nr,dr->nd', coordinates, parts[k]) * geometry['stream_scales'][i] for i, k in enumerate(KEYS)}
    else:
        corrections = {}
        for key in KEYS:
            b = geometry[key + '_basis'][:, :rank].astype('float64')
            z = np.einsum('nd,dr->nr', means[key] - geometry[key + '_center'], b)
            corrections[key] = np.einsum('nr,dr->nd', z, b)
    out = {}
    for key, value in data.items():
        correction = corrections[key][:, None] if value.ndim == 3 else corrections[key]
        x = value.astype('float64') - strength * correction
        out[key] = (np.clip(x, -10, 10) if clip else x).astype('float32')
    return out


class CoupledNormalModel(Model):
    def __init__(self, cfg, dims, n_diseases, geometry=None):
        super().__init__(cfg, dims, n_diseases)
        rank = cfg['normal_rank']; mode = cfg['normal_stream_mode']
        assert rank in [8, 32] and mode in ['independent', 'coupled']
        self.register_buffer('normal_stream_scales', torch.ones(len(KEYS)))
        for key in KEYS:
            self.register_buffer('normal_center_' + key, torch.zeros(dims[key]))
            if mode == 'independent': self.register_buffer('normal_basis_' + key, torch.zeros(dims[key], rank))
        if mode == 'coupled': self.register_buffer('normal_joint_basis', torch.zeros(sum(dims[k] for k in KEYS), rank))
        if geometry is not None:
            self.normal_stream_scales.copy_(torch.as_tensor(geometry['stream_scales']))
            for key in KEYS:
                getattr(self, 'normal_center_' + key).copy_(torch.as_tensor(geometry[key + '_center']))
                if mode == 'independent': getattr(self, 'normal_basis_' + key).copy_(torch.as_tensor(geometry[key + '_basis'][:, :rank]))
            if mode == 'coupled': self.normal_joint_basis.copy_(torch.as_tensor(geometry['joint_basis'][:, :rank]))

    def geometry(self):
        state = {'stream_scales': self.normal_stream_scales.detach().cpu().numpy()}
        for key in KEYS:
            state[key + '_center'] = getattr(self, 'normal_center_' + key).detach().cpu().numpy()
            if self.cfg['normal_stream_mode'] == 'independent': state[key + '_basis'] = getattr(self, 'normal_basis_' + key).detach().cpu().numpy()
        if self.cfg['normal_stream_mode'] == 'coupled': state['joint_basis'] = self.normal_joint_basis.detach().cpu().numpy()
        return state

    def transform(self, x, clip=True):
        strength = self.cfg['normal_strength']
        if strength == 0: return x
        means = {k: v.mean(1) if v.ndim == 3 else v for k, v in x.items()}
        if self.cfg['normal_stream_mode'] == 'coupled':
            stacked = torch.cat([(means[k] - getattr(self, 'normal_center_' + k)) / self.normal_stream_scales[i] for i, k in enumerate(KEYS)], -1)
            correction = (stacked @ self.normal_joint_basis) @ self.normal_joint_basis.T
            parts = correction.split([means[k].shape[-1] for k in KEYS], -1)
            corrections = {k: parts[i] * self.normal_stream_scales[i] for i, k in enumerate(KEYS)}
        else:
            corrections = {}
            for key in KEYS:
                b = getattr(self, 'normal_basis_' + key)
                corrections[key] = ((means[key] - getattr(self, 'normal_center_' + key)) @ b) @ b.T
        out = {}
        for key, value in x.items():
            delta = corrections[key].unsqueeze(1) if value.ndim == 3 else corrections[key]
            transformed = value - strength * delta
            out[key] = transformed.clamp(-10, 10) if clip else transformed
        return out

    def forward(self, x):
        return super().forward(self.transform(x))
