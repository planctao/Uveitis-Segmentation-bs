"""Unit-normalized register attention with fixed cosine temperatures."""
import torch
from torch.nn import functional as F
from large_token_evidence import LargeTokenEvidenceModel, TokenQueryResidual, DETAILS, token_evidence_loss

KINDS = ['dot_control', 'cosine_1', 'cosine_4', 'cosine_8']


class CosineResidual(TokenQueryResidual):
    def __init__(self, kind):
        super().__init__(); assert kind in KINDS; self.kind = kind

    def forward(self, tokens, hidden):
        normalized = self.token_norm(tokens); query = self.query(self.query_norm(hidden))
        keys = self.key(normalized); values = F.gelu(self.value(normalized))
        if self.kind == 'dot_control':
            scores = torch.einsum('bd,btd->bt', query, keys).div(32**.5)
        else:
            query = F.normalize(query, dim=-1, eps=1e-6); keys = F.normalize(keys, dim=-1, eps=1e-6)
            scores = torch.einsum('bd,btd->bt', query, keys)*int(self.kind.rsplit('_', 1)[1])
        attention = scores.softmax(1); pooled = torch.einsum('bt,btd->bd', attention, values)
        delta = .1*self.readout(pooled)
        return {'token_query': query, 'token_keys': keys, 'token_values': values, 'token_attention': attention,
            'token_pooled': pooled, 'token_delta': delta, 'token_base_hidden': hidden}


class RegisterCosineModel(LargeTokenEvidenceModel):
    def __init__(self, cfg, dims, n_diseases, geometry=None, witness_path=None):
        assert cfg['token_evidence_stream'] == 'registers'
        super().__init__(cfg, dims, n_diseases, geometry, witness_path)
        with torch.random.fork_rng(devices=[]): self.token_evidence = CosineResidual(cfg['register_similarity_kind'])
