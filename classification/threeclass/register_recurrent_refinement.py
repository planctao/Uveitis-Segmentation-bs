"""Shared-weight register query refinement at a fixed total residual step length."""
import torch
from large_token_evidence import LargeTokenEvidenceModel, TokenQueryResidual, DETAILS, token_evidence_loss

KINDS = ['control', 'euler2', 'euler4', 'heun']
TRACE = ['refinement_hidden', 'refinement_attention', 'refinement_delta']


class RecurrentRegisterResidual(TokenQueryResidual):
    def __init__(self, kind):
        super().__init__(); assert kind in KINDS; self.kind = kind

    def forward(self, tokens, hidden):
        initial = hidden; trace = []
        def evaluate(value):
            out = super(RecurrentRegisterResidual, self).forward(tokens, value)
            trace.append(out); return out
        if self.kind == 'control':
            out = evaluate(hidden); final = hidden+out['token_delta']; delta = out['token_delta']
        elif self.kind == 'heun':
            first = evaluate(hidden); second = evaluate(hidden+first['token_delta'])
            delta = (first['token_delta']+second['token_delta'])*.5; final = hidden+delta; out = second
        else:
            count = 2 if self.kind == 'euler2' else 4
            for _ in range(count):
                out = evaluate(hidden); hidden = hidden+out['token_delta']/count
            final = hidden; delta = final-initial
        return {**out, 'token_delta':delta, 'token_base_hidden':initial, 'refinement_updated_hidden':final,
            'refinement_hidden':torch.stack([value['token_base_hidden'] for value in trace],1),
            'refinement_attention':torch.stack([value['token_attention'] for value in trace],1),
            'refinement_delta':torch.stack([value['token_delta'] for value in trace],1)}


class RegisterRecurrentRefinementModel(LargeTokenEvidenceModel):
    def __init__(self, cfg, dims, n_diseases, geometry=None, witness_path=None):
        super().__init__(cfg, dims, n_diseases, geometry, witness_path)
        with torch.random.fork_rng(devices=[]): self.token_evidence = RecurrentRegisterResidual(cfg['register_recurrent_refinement'])

    def _add(self, module, args, hidden):
        self._details = self.token_evidence(self._tokens, hidden)
        return self._details['refinement_updated_hidden']
