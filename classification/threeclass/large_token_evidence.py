"""New frozen L token evidence refines the original normal-L hidden representation."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiment import KEYS, loss_fn
from normal_subspace import NormalSuppressedModel, patient_centers

DETAILS = ['token_projected', 'token_query', 'token_keys', 'token_values', 'token_attention', 'token_pooled', 'token_delta', 'token_base_hidden']


def fit_side_geometry(frame, normalized):
    names, centers = patient_centers(frame, {'side_tokens': normalized}, 'side_tokens')
    assert len(names) > 32
    center = centers.mean(0); _, singular, right = np.linalg.svd(centers-center, full_matrices=False); basis = right[:32].T
    signs = np.sign(basis[np.abs(basis).argmax(0), np.arange(32)]); basis *= np.where(signs == 0, 1, signs)
    return {'side_center': center.astype('float32'), 'side_basis': basis.astype('float32'),
        'side_eigenvalues': (singular[:32]**2/(len(names)-1)).astype('float32'), 'side_normal_patients': np.asarray(names, dtype=str)}


class TokenQueryResidual(nn.Module):
    def __init__(self):
        super().__init__(); self.token_norm = nn.LayerNorm(2048); self.query_norm = nn.LayerNorm(256)
        self.query = nn.Linear(256, 32); self.key = nn.Linear(2048, 32); self.value = nn.Linear(2048, 32)
        self.readout = nn.Linear(32, 256); nn.init.zeros_(self.readout.weight); nn.init.zeros_(self.readout.bias)

    def forward(self, tokens, hidden):
        values = self.token_norm(tokens); query = self.query(self.query_norm(hidden)); keys = self.key(values); values = F.gelu(self.value(values))
        attention = torch.einsum('bd,btd->bt', query, keys).div(32**.5).softmax(1)
        pooled = torch.einsum('bt,btd->bd', attention, values); delta = .1*self.readout(pooled)
        return {'token_query': query, 'token_keys': keys, 'token_values': values, 'token_attention': attention,
            'token_pooled': pooled, 'token_delta': delta, 'token_base_hidden': hidden}


class LargeTokenEvidenceModel(NormalSuppressedModel):
    def __init__(self, cfg, dims, n_diseases, geometry=None, witness_path=None):
        super().__init__(cfg, dims, n_diseases, geometry)
        assert all(dims[key] == 2048 for key in KEYS) and dims['side_tokens'] == 2048 and cfg['hidden'] == 256
        self.register_buffer('side_normal_center', torch.zeros(2048)); self.register_buffer('side_normal_basis', torch.zeros(2048, 32))
        if geometry is not None:
            self.side_normal_center.copy_(torch.as_tensor(geometry['side_center'])); self.side_normal_basis.copy_(torch.as_tensor(geometry['side_basis']))
        with torch.random.fork_rng(devices=[]): self.token_evidence = TokenQueryResidual()
        self.encoder[1].register_forward_hook(self._add)
        self.witness_path = None if witness_path is None else Path(witness_path); self.witness_written = False

    def _add(self, module, args, hidden):
        self._details = self.token_evidence(self._tokens, hidden)
        return hidden+self._details['token_delta']

    def forward(self, x):
        context = None
        if self.training and torch.is_grad_enabled() and self.witness_path is not None and not self.witness_written:
            device = x['global'].device
            context = {'model': self, 'raw_global_all': x['global'], 'raw_side_all': x['side_tokens'],
                'rng_cpu': torch.random.get_rng_state().clone(), 'rng_cuda': torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None,
                'initial_parameter_sha256': {name: hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest() for name, p in self.named_parameters()}}
        tokens = x['side_tokens']; self._tokens = (tokens-((tokens-self.side_normal_center)@self.side_normal_basis)@self.side_normal_basis.T).clamp(-10, 10)
        out = super().forward({key: x[key] for key in KEYS}); out.update(self._details, token_projected=self._tokens)
        if context is not None: out['_token_evidence_witness_context'] = context
        return out


def token_evidence_loss(out, y, disease, weights, cfg):
    loss = loss_fn(out, y, disease, weights, cfg); context = out.get('_token_evidence_witness_context')
    if context is not None:
        model = context['model']; gradients = torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=True)
        proof = {'seed': np.array(cfg['seed']), 'batch_size': np.array(len(y)), 'stream': np.array(cfg['token_evidence_stream']),
            'loss': np.array(float(loss.detach())), 'rng_cpu': context['rng_cpu'].numpy()}
        tensors = {'raw_global_all': context['raw_global_all'], 'raw_side_all': context['raw_side_all'], 'y': y, 'disease': disease,
            'probability': out['p'], 'subtype': out['subtype'], 'parameter_gradient': torch.cat([g.flatten() for g in gradients]),
            **{key: out[key] for key in DETAILS}, **{'weight_'+key: value for key, value in weights.items()}}
        for key, value in tensors.items(): proof[key] = value.detach().cpu().numpy()
        if context['rng_cuda'] is not None: proof['rng_cuda'] = context['rng_cuda'].cpu().numpy()
        for name, value in context['initial_parameter_sha256'].items(): proof['initial_parameter_sha256_'+name] = np.array(value)
        model.witness_path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(model.witness_path, **proof); model.witness_written = True
    return loss
