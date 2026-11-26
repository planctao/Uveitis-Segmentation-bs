"""Select register evidence independently in the projected scale channel blocks."""
import torch
from torch.nn import functional as F
from large_token_evidence import LargeTokenEvidenceModel, TokenQueryResidual, DETAILS, token_evidence_loss

KINDS = ['legacy_control', 'factored_shared', 'factored_independent', 'factored_scale_norm']


class ScaleFactorizedResidual(TokenQueryResidual):
    def __init__(self, kind):
        super().__init__(); assert kind in KINDS; self.kind = kind

    def forward(self, tokens, hidden):
        if self.kind == 'legacy_control': return super().forward(tokens, hidden)
        if self.kind == 'factored_scale_norm':
            normalized = F.layer_norm(tokens.reshape(*tokens.shape[:-1], 2, 1024), (1024,)).flatten(-2)
            normalized = normalized*self.token_norm.weight+self.token_norm.bias
        else: normalized = self.token_norm(tokens)
        query = self.query(self.query_norm(hidden))
        keys = torch.stack([F.linear(normalized[...,s*1024:(s+1)*1024], self.key.weight[:,s*1024:(s+1)*1024], self.key.bias/2) for s in range(2)], 1)
        values = torch.stack([F.gelu(F.linear(normalized[...,s*1024:(s+1)*1024], self.value.weight[:,s*1024:(s+1)*1024], self.value.bias/2)) for s in range(2)], 1)
        if self.kind == 'factored_shared':
            keys = keys.sum(1)
            attention = torch.einsum('bd,btd->bt', query, keys).div(32**.5).softmax(-1)
            pooled = torch.einsum('bt,bstd->bd', attention, values)
        else:
            attention = torch.einsum('bd,bstd->bst', query, keys).div(32**.5).softmax(-1)
            pooled = torch.einsum('bst,bstd->bd', attention, values)
        delta = .1*self.readout(pooled)
        return {'token_query': query, 'token_keys': keys, 'token_values': values, 'token_attention': attention,
            'token_pooled': pooled, 'token_delta': delta, 'token_base_hidden': hidden}


class RegisterScaleFactorizationModel(LargeTokenEvidenceModel):
    def __init__(self, cfg, dims, n_diseases, geometry=None, witness_path=None):
        assert cfg['token_evidence_stream'] == 'registers'
        super().__init__(cfg, dims, n_diseases, geometry, witness_path)
        with torch.random.fork_rng(devices=[]): self.token_evidence = ScaleFactorizedResidual(cfg['register_scale_kind'])
