"""Normal-subspace training augmentation with coherent or Gaussian nuisance draws."""
from pathlib import Path
import numpy as np
import torch
from experiment import KEYS
from normal_subspace import NormalSuppressedModel, patient_centers


def augmentation_geometry(frame, normalized, geometry):
    result = dict(geometry)
    for key in KEYS:
        names, centers = patient_centers(frame, normalized, key)
        assert np.array_equal(names, geometry['normal_patients'])
        result[key + '_coordinates'] = ((centers - geometry[key + '_center'].astype('float64')) @
                                        geometry[key + '_basis'].astype('float64')).astype('float32')
    return result


def draw_coordinates(count, mode, banks, deviations, rng):
    assert mode in ['coherent', 'gaussian']
    n = len(banks['global'])
    assert n > 1 and all(len(banks[key]) == n for key in KEYS)
    if mode == 'coherent':
        first = rng.integers(0, n, count)
        second = (first + rng.integers(1, n, count)) % n
        coordinates = {key: ((banks[key][first] - banks[key][second]) / np.sqrt(2)).astype('float32') for key in KEYS}
        return coordinates, {'first': first, 'second': second}
    return {key: (rng.standard_normal((count, banks[key].shape[1])) * deviations[key]).astype('float32')
            for key in KEYS}, {}


class NormalAxisAugmentedModel(NormalSuppressedModel):
    def __init__(self, cfg, dims, n_diseases, geometry=None, witness_path=None):
        super().__init__(cfg, dims, n_diseases, geometry)
        assert cfg['normal_rank'] == 32 and cfg['normal_strength'] == 1.
        assert cfg['normal_axis_mode'] in ['coherent', 'gaussian']
        assert 0 <= cfg['normal_axis_scale'] <= .5
        count = dims.get('normal_axis_patients', len(geometry['normal_patients']) if geometry is not None else 0)
        assert count > 1
        dims['normal_axis_patients'] = count
        self.cfg = cfg
        self.axis_rng = np.random.default_rng(20281524 + cfg.get('seed', 0))
        self.witness_path = None if witness_path is None else Path(witness_path)
        self.witness_written = False
        for key in KEYS:
            self.register_buffer('axis_coordinates_' + key, torch.zeros(count, 32))
            self.register_buffer('axis_deviation_' + key, torch.zeros(32))
            if geometry is not None:
                getattr(self, 'axis_coordinates_' + key).copy_(torch.as_tensor(geometry[key + '_coordinates']))
                getattr(self, 'axis_deviation_' + key).copy_(torch.as_tensor(geometry[key + '_eigenvalues']).sqrt())

    def transform(self, x):
        base = super().transform(x)
        scale = self.cfg['normal_axis_scale']
        if not self.training or not scale:
            return base
        banks = {key: getattr(self, 'axis_coordinates_' + key).detach().cpu().numpy() for key in KEYS}
        deviations = {key: getattr(self, 'axis_deviation_' + key).detach().cpu().numpy() for key in KEYS}
        coordinates, donors = draw_coordinates(len(x['global']), self.cfg['normal_axis_mode'], banks, deviations, self.axis_rng)
        result = {}
        for key, value in base.items():
            coef = torch.as_tensor(coordinates[key], device=value.device, dtype=value.dtype)
            delta = coef @ getattr(self, 'normal_basis_' + key).T
            if value.ndim == 3:
                delta = delta[:, None, :]
            result[key] = (value + scale * delta).clamp(-10, 10)
        if self.witness_path is not None and not self.witness_written:
            proof = {'batch_size': np.array(len(x['global'])), 'scale': np.array(scale),
                     'mode': np.array(self.cfg['normal_axis_mode']), 'seed': np.array(self.cfg['seed'])}
            for key in KEYS:
                proof['input_' + key] = x[key][:8].detach().cpu().numpy()
                proof['output_' + key] = result[key][:8].detach().cpu().numpy()
                proof['draw_' + key] = coordinates[key][:8]
            proof.update({name: ids[:8] for name, ids in donors.items()})
            self.witness_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.witness_path, **proof)
            self.witness_written = True
        return result
