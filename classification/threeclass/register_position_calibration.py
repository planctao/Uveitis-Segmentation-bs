"""Patient-balanced position moments remove fixed register-identity offsets."""
import numpy as np
import torch
from large_token_evidence import LargeTokenEvidenceModel, token_evidence_loss
from normal_subspace import independent_transform


def fit_position_moments(frame, projected, scope):
    assert scope in ['normal', 'all'] and projected.shape == (len(frame), 4, 2048)
    mask = frame.label.eq(2).to_numpy() if scope == 'normal' else np.ones(len(frame), bool)
    names = sorted(frame.loc[mask, 'patient_group'].unique()); first = []; second = []
    for name in names:
        values = projected[mask & frame.patient_group.eq(name).to_numpy()].astype('float64')
        first.append(values.mean(0)); second.append(np.square(values).mean(0))
    mean = np.mean(first, 0); variance = np.maximum(np.mean(second, 0)-mean**2, 0)
    return {'position_mean': mean.astype('float32'), 'position_scale': np.sqrt(np.maximum(variance, .05**2)).astype('float32'),
        'position_global_mean': mean.mean(0, keepdims=True).astype('float32'), 'position_patients': np.asarray(names, dtype=str)}


class RegisterPositionCalibrationModel(LargeTokenEvidenceModel):
    def __init__(self, cfg, dims, n_diseases, geometry=None, witness_path=None):
        assert cfg['token_evidence_stream'] == 'registers'
        assert cfg['register_position_kind'] in ['none', 'normal_mean', 'normal_standardize', 'all_standardize']
        super().__init__(cfg, dims, n_diseases, geometry, witness_path)
        self.register_buffer('position_mean', torch.zeros(4, 2048)); self.register_buffer('position_scale', torch.ones(4, 2048))
        self.register_buffer('position_global_mean', torch.zeros(1, 2048))
        if geometry is not None:
            for name in ['position_mean', 'position_scale', 'position_global_mean']:
                getattr(self, name).copy_(torch.as_tensor(geometry[name]))

    def _add(self, module, args, hidden):
        kind = self.cfg['register_position_kind']; tokens = self._tokens
        if kind == 'normal_mean':
            tokens = (tokens-(self.position_mean-self.position_global_mean)).clamp(-10, 10)
        elif kind in ['normal_standardize', 'all_standardize']:
            tokens = ((tokens-self.position_mean)/self.position_scale+self.position_global_mean).clamp(-10, 10)
        self._tokens = tokens; self._details = self.token_evidence(tokens, hidden)
        return hidden+self._details['token_delta']
