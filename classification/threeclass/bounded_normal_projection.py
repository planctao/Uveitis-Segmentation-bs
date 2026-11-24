"""Remove a bounded normal-axis component while retaining large coordinates."""
import numpy as np
import torch
from experiment import KEYS
from normal_subspace import NormalSuppressedModel


def independent_bounded_transform(values, center, basis, deviation, factor, shape):
    x = np.asarray(values, dtype='float64')
    if factor == 0:
        return x.astype('float32')
    b = np.asarray(basis, dtype='float64')
    z = np.einsum('...d,dk->...k', x - np.asarray(center, dtype='float64'), b)
    limit = factor * np.asarray(deviation, dtype='float64')
    if shape == 'hard':
        removed = np.minimum(np.maximum(z, -limit), limit)
    else:
        assert shape == 'smooth'
        removed = limit * np.tanh(z / np.maximum(limit, 1e-8))
    return np.clip(x - np.einsum('...k,dk->...d', removed, b), -10, 10).astype('float32')


class BoundedNormalModel(NormalSuppressedModel):
    def __init__(self, cfg, dims, n_diseases, geometry=None):
        super().__init__(cfg, dims, n_diseases, geometry)
        assert cfg['normal_rank'] == 32 and cfg['normal_strength'] == 1.
        assert cfg['normal_bound_shape'] in ['hard', 'smooth'] and cfg['normal_bound_factor'] >= 0
        for key in KEYS:
            self.register_buffer('normal_deviation_' + key, torch.zeros(32))
            if geometry is not None:
                getattr(self, 'normal_deviation_' + key).copy_(torch.as_tensor(geometry[key + '_eigenvalues']).sqrt())

    def transform(self, x):
        factor = self.cfg['normal_bound_factor']
        if factor == 0:
            return x
        result = {}
        for key, value in x.items():
            basis = getattr(self, 'normal_basis_' + key)
            z = (value - getattr(self, 'normal_center_' + key)) @ basis
            limit = factor * getattr(self, 'normal_deviation_' + key)
            if self.cfg['normal_bound_shape'] == 'hard':
                removed = torch.minimum(torch.maximum(z, -limit), limit)
            else:
                removed = limit * torch.tanh(z / limit.clamp(min=1e-8))
            result[key] = (value - removed @ basis.T).clamp(-10, 10)
        return result
