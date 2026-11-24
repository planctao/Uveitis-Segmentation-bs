"""A global-conditioned, class-specific local evidence pooling candidate.

Background validity comes from image brightness only, never from class labels.
The sparse evidence path complements attention pooling and the global feature.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

class EvidencePool(nn.Module):
    def __init__(self,dims,cfg):
        super().__init__();self.cfg=cfg;d=128
        self.region_proj=nn.Sequential(nn.Linear(dims['regions'],d),nn.LayerNorm(d),nn.GELU())
        self.class_query=nn.Parameter(torch.randn(3,d)*.02)
        self.global_query=nn.Linear(dims['global'],3*d)
        self.class_weight=nn.Parameter(torch.randn(3,d)*.02)
        self.class_bias=nn.Parameter(torch.zeros(3))
        self.score_scale=nn.Parameter(torch.tensor(math.log(5.)))
        self.global_proj=nn.Linear(dims['global'],256)
        self.mid_proj=nn.Linear(dims['mid'],64)
        self.std_proj=nn.Linear(dims['dispersion'],64)

    def forward(self,x):
        tokens=self.region_proj(x['regions']);n,k,d=tokens.shape
        valid=torch.ones(n,k,dtype=torch.bool,device=tokens.device)
        if 'region_mask' in x and self.cfg.get('use_region_mask',True):valid=x['region_mask']>.5
        valid=torch.where(valid.any(1,keepdim=True),valid,torch.ones_like(valid))
        drop=self.cfg.get('region_dropout',0.)
        if self.training and drop:
            retained=valid & (torch.rand_like(valid.float())>=drop)
            valid=torch.where(retained.any(1,keepdim=True),retained,valid)
        query=self.class_query[None].expand(n,-1,-1)
        if self.cfg.get('global_conditioning',True):query=query+self.global_query(x['global']).reshape(n,3,d)
        scale=self.score_scale.clamp(0,math.log(30.)).exp()
        attention_scores=torch.einsum('bkd,bcd->bkc',F.normalize(tokens,dim=-1),F.normalize(query,dim=-1))*scale
        attention=attention_scores.masked_fill(~valid[:,:,None],-1e4).softmax(1)
        pooled=torch.einsum('bkc,bkd->bcd',attention,tokens)
        region_scores=torch.einsum('bkd,cd->bkc',F.normalize(tokens,dim=-1),F.normalize(self.class_weight,dim=-1))*scale
        sparse=region_scores.masked_fill(~valid[:,:,None],-1e4).topk(min(self.cfg.get('topk',5),k),dim=1).values
        retained=sparse>-5000
        top_score=(sparse*retained).sum(1)/retained.sum(1).clamp(min=1)
        mean_score=(attention*region_scores).sum(1)
        local_logits=.5*(top_score+mean_score)+self.class_bias
        representation=torch.cat([F.gelu(self.global_proj(x['global'])),pooled.flatten(1),F.gelu(self.mid_proj(x['mid'])),F.gelu(self.std_proj(x['dispersion']))],-1)
        return representation,local_logits
