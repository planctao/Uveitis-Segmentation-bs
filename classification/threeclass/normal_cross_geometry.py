"""Normal-patient cross-backbone covariance directions with independent PCA controls."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from experiment import KEYS, Model
from normal_subspace import patient_centers, independent_transform


def fit_geometry(frame, normalized, kind, rank):
    assert kind in ['independent', 'shared'] and rank in [8, 32]
    geometry = {}
    for key in KEYS:
        names, base = patient_centers(frame, {key: normalized[key][..., :1536]}, key)
        other_names, large = patient_centers(frame, {key: normalized[key][..., 1536:]}, key)
        assert names == other_names and len(names) > rank
        base_mean, large_mean = base.mean(0), large.mean(0)
        a, b = base-base_mean, large-large_mean
        if kind == 'independent':
            _, sa, va = np.linalg.svd(a, full_matrices=False)
            _, sb, vb = np.linalg.svd(b, full_matrices=False)
            u, v = va[:rank].T, vb[:rank].T
            for basis in [u, v]:
                signs = np.sign(basis[np.abs(basis).argmax(0), np.arange(rank)])
                basis *= np.where(signs == 0, 1, signs)
            spectrum = np.r_[sa[:rank]**2/(len(names)-1), sb[:rank]**2/(len(names)-1)]
        else:
            qa, ra = np.linalg.qr(a.T, mode='reduced')
            qb, rb = np.linalg.qr(b.T, mode='reduced')
            left, spectrum, right = np.linalg.svd(ra@rb.T/(len(names)-1), full_matrices=False)
            u, v = qa@left[:, :rank], qb@right[:rank].T
            signs = np.sign(u[np.abs(u).argmax(0), np.arange(rank)])
            signs = np.where(signs == 0, 1, signs)
            u *= signs
            v *= signs
            spectrum = spectrum[:rank]
        for prefix, center, basis in [('base_', base_mean, u), ('large_', large_mean, v)]:
            geometry[prefix+key+'_center'] = center.astype('float32')
            geometry[prefix+key+'_basis'] = basis.astype('float32')
        geometry[key+'_spectrum'] = spectrum.astype('float32')
    geometry['normal_patients'] = np.asarray(names, dtype=str)
    return geometry


class NormalCrossGeometryModel(Model):
    def __init__(self, cfg, dims, n_diseases, geometry=None, witness_path=None):
        super().__init__(cfg, dims, n_diseases)
        assert all(dims[k] == 3584 for k in KEYS)
        rank = cfg['cross_geometry_rank']
        assert rank in [8, 32]
        self.witness_path = None if witness_path is None else Path(witness_path)
        self.witness_written = False
        for prefix, dimension in [('base_', 1536), ('large_', 2048)]:
            for key in KEYS:
                self.register_buffer(prefix+'center_'+key, torch.zeros(dimension))
                self.register_buffer(prefix+'basis_'+key, torch.zeros(dimension, rank))
                if geometry is not None:
                    getattr(self, prefix+'center_'+key).copy_(torch.as_tensor(geometry[prefix+key+'_center']))
                    getattr(self, prefix+'basis_'+key).copy_(torch.as_tensor(geometry[prefix+key+'_basis']))

    def transform(self, x):
        result = {}
        for key, value in x.items():
            parts = []
            for prefix, section in [('base_', slice(None, 1536)), ('large_', slice(1536, None))]:
                z = value[..., section]
                center = getattr(self, prefix+'center_'+key)
                basis = getattr(self, prefix+'basis_'+key)
                parts.append((z-((z-center)@basis)@basis.T).clamp(-10, 10))
            result[key] = torch.cat(parts, dim=-1)
        return result

    def forward(self, x):
        context = None
        if self.training and torch.is_grad_enabled() and self.witness_path is not None and not self.witness_written:
            device = x['global'].device
            context = {'model': self, 'raw_global_all': x['global'], 'rng_cpu': torch.random.get_rng_state().clone(),
                'rng_cuda': torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None,
                'initial_parameter_sha256': {k: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k, v in self.named_parameters()}}
        projected = self.transform(x)
        out = super().forward(projected)
        if context is not None:
            context.update(projected_global=projected['global'], projected_region_mean=projected['regions'].mean(1))
            out['_large_normal_witness_context'] = context
        return out


def independent_features(raw, state):
    result = {}
    for key, value in raw.items():
        result[key] = np.concatenate([independent_transform(value[..., section],
            state[prefix+'center_'+key].detach().cpu().numpy(),
            state[prefix+'basis_'+key].detach().cpu().numpy(), 1.)
            for prefix, section in [('base_', slice(None, 1536)), ('large_', slice(1536, None))]], axis=-1)
    return result
