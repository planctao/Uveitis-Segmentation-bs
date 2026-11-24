"""Disease-specific region queries supply gated residual subtype evidence."""
import torch
from torch import nn
import torch.nn.functional as F


class SubtypeEvidenceQueries(nn.Module):
    def __init__(self,diseases,dimension,cfg):
        super().__init__();self.cfg=cfg
        heads=cfg.get('query_heads',2)
        self.queries=nn.Parameter(torch.randn(diseases,heads,dimension)*.02)
        self.readout=nn.Parameter(torch.zeros(diseases,heads,dimension))
        self.bias=nn.Parameter(torch.zeros(diseases))
        self.temperature=cfg.get('query_temperature',.2)
        if self.temperature<=0:raise ValueError('Query temperature must be positive')

    def forward(self,tokens,mask=None):
        values=F.dropout(tokens,p=self.cfg.get('query_dropout',.1),training=self.training)
        scores=torch.einsum('blr,dhr->bdhl',F.normalize(values,dim=-1),F.normalize(self.queries,dim=-1))/self.temperature
        if mask is not None:
            valid=mask>.5
            valid=torch.where(valid.any(1,keepdim=True),valid,torch.ones_like(valid))
            scores=scores.masked_fill(~valid[:,None,None,:],-1e4)
        attention=scores.softmax(-1)
        pooled=torch.einsum('bdhl,blr->bdhr',attention,values)
        evidence=torch.einsum('bdhr,dhr->bdh',pooled,self.readout).mean(-1)+self.bias
        return evidence,attention
