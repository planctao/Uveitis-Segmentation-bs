"""Separate image-specific register evidence from a fixed patient-balanced context bank."""
import torch
from torch.nn import functional as F
from large_token_evidence import LargeTokenEvidenceModel, TokenQueryResidual, DETAILS, token_evidence_loss
from register_position_calibration import fit_position_moments

KINDS = ['real_control', 'static_context', 'dynamic_deviation', 'static_dynamic']


class StaticContextResidual(TokenQueryResidual):
    def __init__(self, kind):
        super().__init__(); assert kind in KINDS; self.kind = kind

    def forward(self, tokens, hidden, bank):
        static = bank[None].expand(len(tokens), -1, -1)
        if self.kind == 'real_control': return super().forward(tokens, hidden)
        if self.kind == 'static_context': return super().forward(static, hidden)
        dynamic = F.layer_norm(tokens-static, (2048,), self.token_norm.weight, self.token_norm.bias, eps=.05**2)
        normalized = dynamic if self.kind == 'dynamic_deviation' else torch.stack([self.token_norm(static), dynamic], 1)
        query = self.query(self.query_norm(hidden)); keys = self.key(normalized); values = F.gelu(self.value(normalized))
        if self.kind == 'dynamic_deviation':
            attention = torch.einsum('bd,btd->bt', query, keys).div(32**.5).softmax(-1)
            pooled = torch.einsum('bt,btd->bd', attention, values)
        else:
            attention = torch.einsum('bd,bstd->bst', query, keys).div(32**.5).softmax(-1)
            pooled = torch.einsum('bst,bstd->bsd', attention, values).mean(1)
        delta = .1*self.readout(pooled)
        return {'token_query': query, 'token_keys': keys, 'token_values': values, 'token_attention': attention,
            'token_pooled': pooled, 'token_delta': delta, 'token_base_hidden': hidden}


class RegisterStaticContextModel(LargeTokenEvidenceModel):
    def __init__(self, cfg, dims, n_diseases, geometry=None, witness_path=None):
        assert cfg['token_evidence_stream'] == 'registers'
        super().__init__(cfg, dims, n_diseases, geometry, witness_path)
        self.register_buffer('position_mean', torch.zeros(4, 2048))
        if geometry is not None: self.position_mean.copy_(torch.as_tensor(geometry['position_mean']))
        with torch.random.fork_rng(devices=[]): self.token_evidence = StaticContextResidual(cfg['register_context_kind'])

    def _add(self, module, args, hidden):
        self._details = self.token_evidence(self._tokens, hidden, self.position_mean)
        return hidden+self._details['token_delta']
