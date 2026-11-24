"""Partial pooling of subtype logits around shared etiology classifiers."""
import torch
from torch import nn

class SupportResidualHead(nn.Module):
    def __init__(self,dimension,mapping,counts,cfg):
        super().__init__();self.cfg=cfg
        mapping=torch.as_tensor(mapping,dtype=torch.long);counts=torch.as_tensor(counts,dtype=torch.float32)
        if len(mapping)!=len(counts) or set(mapping.tolist())!={0,1,2} or not (counts>0).all():
            raise ValueError('Subtype support must contain positive fit-patient counts and all three families')
        self.coarse=nn.Linear(dimension,3);self.residual=nn.Linear(dimension,len(mapping))
        nn.init.zeros_(self.residual.weight);nn.init.zeros_(self.residual.bias)
        strength=float(cfg.get('support_shrink',10.))
        if strength<0:raise ValueError('Support shrinkage must be nonnegative')
        prior_kind=cfg.get('support_prior','support')
        if prior_kind not in ['support','uniform']:raise ValueError('Unknown subtype support prior')
        prior=torch.ones_like(counts) if prior_kind=='uniform' else counts+float(cfg.get('support_alpha',1.))
        for family in range(3):
            mask=mapping==family;prior[mask]/=prior[mask].sum()
        self.register_buffer('mapping',mapping);self.register_buffer('support',counts)
        self.register_buffer('prior',prior);self.register_buffer('log_prior',prior.log())
        self.register_buffer('shrinkage',(counts/(counts+strength)).sqrt())
        self.register_buffer('penalty_weight',1+strength/counts)

    def centered_residual(self,x):
        raw=self.residual(x)*self.shrinkage
        centers=torch.stack([(raw[:,self.mapping==c]*self.prior[self.mapping==c]).sum(1) for c in range(3)],1)
        return raw-centers[:,self.mapping]

    def forward(self,x):
        return self.coarse(x)[:,self.mapping]+self.centered_residual(x)+self.log_prior

    def regularization(self):
        per_subtype=self.residual.weight.square().mean(1)+self.residual.bias.square()
        return (per_subtype*self.penalty_weight).mean()
