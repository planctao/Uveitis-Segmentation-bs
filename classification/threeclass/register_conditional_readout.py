"""Apply the same register readout additively or as hidden-channel modulation."""
import torch
from torch.nn import functional as F
from large_token_evidence import LargeTokenEvidenceModel, TokenQueryResidual, DETAILS
from compact_token_witness import token_evidence_loss

KINDS = ['control', 'bounded_add', 'multiplicative', 'hybrid']
TRACE = ['conditional_readout', 'conditional_normalized_hidden', 'conditional_factor']


class ConditionalRegisterResidual(TokenQueryResidual):
    def __init__(self, kind):
        super().__init__(); assert kind in KINDS; self.kind = kind

    def forward(self, tokens, hidden):
        result = super().forward(tokens,hidden)
        readout = self.readout(result['token_pooled'])
        normalized = F.layer_norm(hidden,(256,),eps=1e-5)
        factor = torch.ones_like(hidden)
        if self.kind == 'multiplicative': factor = normalized
        elif self.kind == 'hybrid': factor = (1+normalized)/(2**.5)
        if self.kind != 'control': result['token_delta'] = .1*readout.tanh()*factor
        return {**result,'conditional_readout':readout,'conditional_normalized_hidden':normalized,'conditional_factor':factor}


class RegisterConditionalReadoutModel(LargeTokenEvidenceModel):
    def __init__(self,cfg,dims,n_diseases,geometry=None,witness_path=None):
        super().__init__(cfg,dims,n_diseases,geometry,witness_path)
        with torch.random.fork_rng(devices=[]): self.token_evidence=ConditionalRegisterResidual(cfg['register_conditional_readout'])
