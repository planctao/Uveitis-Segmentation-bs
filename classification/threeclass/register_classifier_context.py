"""Use existing subtype-classifier weights as register-query contexts."""
import torch
from torch.nn import functional as F
from large_token_evidence import LargeTokenEvidenceModel, TokenQueryResidual, DETAILS, token_evidence_loss

KINDS = ['hidden_control', 'class_linked', 'class_detached', 'disease_balanced']


class ClassifierContextResidual(TokenQueryResidual):
    def __init__(self, kind):
        super().__init__(); assert kind in KINDS; self.kind = kind

    def forward(self, tokens, hidden, classifier, mapping):
        if self.kind == 'hidden_control': return super().forward(tokens, hidden)
        counts = mapping.sum(0)
        if self.kind == 'disease_balanced':
            contexts = classifier
            query_weight = (mapping/counts).sum(1)/3
        else:
            contexts = (mapping.T@classifier)/counts[:,None]
            if self.kind == 'class_detached': contexts = contexts.detach()
            query_weight = torch.full((3,), 1/3, device=hidden.device, dtype=hidden.dtype)
        contexts = F.layer_norm(contexts, (256,))
        query = self.query(self.query_norm(hidden)[:,None]+contexts[None])
        normalized = self.token_norm(tokens); keys = self.key(normalized); values = F.gelu(self.value(normalized))
        attention = torch.einsum('bqd,btd->bqt', query, keys).div(32**.5).softmax(-1)
        per_query = torch.einsum('bqt,btd->bqd', attention, values)
        pooled = (per_query*query_weight[None,:,None]).sum(1); delta = .1*self.readout(pooled)
        return {'token_query': query, 'token_keys': keys, 'token_values': values, 'token_attention': attention,
            'token_pooled': pooled, 'token_delta': delta, 'token_base_hidden': hidden}


class RegisterClassifierContextModel(LargeTokenEvidenceModel):
    def __init__(self, cfg, dims, n_diseases, geometry=None, witness_path=None):
        super().__init__(cfg, dims, n_diseases, geometry, witness_path)
        with torch.random.fork_rng(devices=[]): self.token_evidence = ClassifierContextResidual(cfg['register_classifier_context'])

    def _add(self, module, args, hidden):
        self._details = self.token_evidence(self._tokens, hidden, self.flat.weight, self.subtype_to_class)
        return hidden+self._details['token_delta']
