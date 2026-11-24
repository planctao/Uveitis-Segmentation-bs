"""Three latent hidden-conditioned register queries with a tied-query control."""
import torch
from torch import nn
from torch.nn import functional as F
from large_token_evidence import LargeTokenEvidenceModel, DETAILS, token_evidence_loss


class MultiQueryResidual(nn.Module):
    def __init__(self, rank, kind):
        super().__init__(); assert rank in [16, 32] and kind in ['tied', 'multi']
        self.rank = rank; self.kind = kind
        self.token_norm = nn.LayerNorm(2048); self.query_norm = nn.LayerNorm(256)
        self.query = nn.Linear(256, 3*rank); self.key = nn.Linear(2048, rank); self.value = nn.Linear(2048, rank)
        self.readout = nn.Linear(3*rank, 256); nn.init.zeros_(self.readout.weight); nn.init.zeros_(self.readout.bias)

    def forward(self, tokens, hidden):
        normalized = self.token_norm(tokens); keys = self.key(normalized); values = F.gelu(self.value(normalized))
        query = self.query(self.query_norm(hidden)).reshape(-1, 3, self.rank)
        if self.kind == 'tied': query = query.mean(1, keepdim=True).expand(-1, 3, -1)
        attention = torch.einsum('bqd,btd->bqt', query, keys).div(self.rank**.5).softmax(-1)
        pooled = torch.einsum('bqt,btd->bqd', attention, values)
        delta = .1*self.readout(pooled.flatten(1))
        return {'token_query': query, 'token_keys': keys, 'token_values': values, 'token_attention': attention,
            'token_pooled': pooled, 'token_delta': delta, 'token_base_hidden': hidden}


class RegisterMultiqueryModel(LargeTokenEvidenceModel):
    def __init__(self, cfg, dims, n_diseases, geometry=None, witness_path=None):
        assert cfg['token_evidence_stream'] == 'registers'
        super().__init__(cfg, dims, n_diseases, geometry, witness_path)
        with torch.random.fork_rng(devices=[]):
            self.token_evidence = MultiQueryResidual(cfg['register_query_rank'], cfg['register_query_kind'])
