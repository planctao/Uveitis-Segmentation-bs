"""Parameter-matched shared versus differential register evidence."""
import torch
from torch.nn import functional as F
from large_token_evidence import LargeTokenEvidenceModel, TokenQueryResidual, DETAILS, token_evidence_loss

KINDS = ['attention_control', 'uniform_mean', 'centered_values', 'centered_tokens']


class CommonModeResidual(TokenQueryResidual):
    def __init__(self, kind):
        super().__init__(); assert kind in KINDS; self.kind = kind

    def forward(self, tokens, hidden):
        if self.kind == 'centered_tokens': tokens = tokens-tokens.mean(1, keepdim=True)
        normalized = self.token_norm(tokens); query = self.query(self.query_norm(hidden))
        keys = self.key(normalized); values = F.gelu(self.value(normalized))
        scores = torch.einsum('bd,btd->bt', query, keys).div(32**.5)
        if self.kind == 'uniform_mean': scores = 0.*scores
        attention = scores.softmax(1)
        pooled = torch.einsum('bt,btd->bd', attention, values)
        if self.kind == 'centered_values': pooled = pooled-values.mean(1)
        delta = .1*self.readout(pooled)
        return {'token_query': query, 'token_keys': keys, 'token_values': values, 'token_attention': attention,
            'token_pooled': pooled, 'token_delta': delta, 'token_base_hidden': hidden}


class RegisterCommonModeModel(LargeTokenEvidenceModel):
    def __init__(self, cfg, dims, n_diseases, geometry=None, witness_path=None):
        assert cfg['token_evidence_stream'] == 'registers'
        super().__init__(cfg, dims, n_diseases, geometry, witness_path)
        with torch.random.fork_rng(devices=[]): self.token_evidence = CommonModeResidual(cfg['register_pool_kind'])
