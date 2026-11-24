"""Structured normal-patient register covariance before hidden query pooling."""
import numpy as np
import torch
from experiment import KEYS
from large_token_evidence import LargeTokenEvidenceModel, DETAILS, token_evidence_loss


def fit_pca(rows, rank):
    assert len(rows) > rank
    center = rows.mean(0); _, singular, right = np.linalg.svd(rows-center, full_matrices=False); basis = right[:rank].T
    sign = np.sign(basis[np.abs(basis).argmax(0), np.arange(rank)]); basis *= np.where(sign == 0, 1, sign)
    return center.astype('float32'), basis.astype('float32'), (singular[:rank]**2/(len(rows)-1)).astype('float32')


def fit_register_geometry(frame, tokens):
    assert tokens.shape == (len(frame), 4, 2048)
    normal = frame.label.eq(2).to_numpy(); names = sorted(frame.loc[normal, 'patient_group'].unique())
    centers = np.stack([tokens[normal & frame.patient_group.eq(name).to_numpy()].astype('float64').mean(0) for name in names])
    center, basis, eigen = fit_pca(centers.reshape(len(names), -1), 32)
    position = [fit_pca(centers[:, i], 8) for i in range(4)]
    return {'register_patients': np.asarray(names, dtype=str), 'register_joint_center': center, 'register_joint_basis': basis,
        'register_joint_eigenvalues': eigen, 'register_position_center': np.stack([x[0] for x in position]),
        'register_position_basis': np.stack([x[1] for x in position]), 'register_position_eigenvalues': np.stack([x[2] for x in position])}


class RegisterGeometryModel(LargeTokenEvidenceModel):
    def __init__(self, cfg, dims, n_diseases, geometry=None, witness_path=None):
        assert cfg['token_evidence_stream'] == 'registers' and cfg['register_geometry_kind'] in ['none', 'pooled', 'joint', 'position']
        super().__init__(cfg, dims, n_diseases, geometry, witness_path)
        self.register_buffer('register_joint_center', torch.zeros(8192)); self.register_buffer('register_joint_basis', torch.zeros(8192, 32))
        self.register_buffer('register_position_center', torch.zeros(4, 2048)); self.register_buffer('register_position_basis', torch.zeros(4, 2048, 8))
        if geometry is not None:
            for name in ['register_joint_center', 'register_joint_basis', 'register_position_center', 'register_position_basis']:
                getattr(self, name).copy_(torch.as_tensor(geometry[name]))

    def forward(self, x):
        # Parent owns the original task, CUDA input witness and normalL projection.
        self._raw_registers = x['side_tokens']
        return super().forward(x)

    def _add(self, module, args, hidden):
        kind = self.cfg['register_geometry_kind']; tokens = self._raw_registers
        if kind == 'none': self._tokens = tokens.clamp(-10, 10)
        elif kind == 'joint':
            value = tokens.flatten(1); basis = self.register_joint_basis
            self._tokens = (value-((value-self.register_joint_center)@basis)@basis.T).reshape_as(tokens).clamp(-10, 10)
        elif kind == 'position':
            basis = self.register_position_basis; difference = tokens-self.register_position_center
            coordinates = torch.einsum('btd,tdr->btr', difference, basis)
            self._tokens = (tokens-torch.einsum('btr,tdr->btd', coordinates, basis)).clamp(-10, 10)
        self._details = self.token_evidence(self._tokens, hidden)
        return hidden+self._details['token_delta']
