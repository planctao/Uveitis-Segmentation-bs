"""Shared regional encoders and image-dependent fusion of two DINO scales."""
import torch
from torch import nn
import torch.nn.functional as F

class ScaleConsensusPool(nn.Module):
    def __init__(self,dims,cfg):
        super().__init__();self.cfg=cfg
        if any(dims[k]%2 for k in ['global','regions','mid','dispersion']):
            raise ValueError('Two-scale features require two equal channel partitions')
        heads=cfg.get('region_heads',2);self.dimension=384+128*heads
        self.global_proj=nn.Linear(dims['global']//2,256)
        self.region_proj=nn.Linear(dims['regions']//2,128)
        self.region_gate=nn.Sequential(nn.Linear(128,64),nn.Tanh(),nn.Linear(64,heads))
        self.mid_proj=nn.Linear(dims['mid']//2,64)
        self.std_proj=nn.Linear(dims['dispersion']//2,64)
        self.reliability=nn.Sequential(nn.Linear(self.dimension,64),nn.GELU(),nn.Linear(64,1))
        nn.init.zeros_(self.reliability[-1].weight);nn.init.zeros_(self.reliability[-1].bias)

    def forward(self,x):
        halves={k:x[k].chunk(2,dim=-1) for k in ['global','regions','mid','dispersion']}
        views=[]
        for index in range(2):
            regions=F.gelu(self.region_proj(halves['regions'][index]))
            scores=self.region_gate(regions)
            if 'region_mask' in x:
                valid=x['region_mask']>.5
                valid=torch.where(valid.any(1,keepdim=True),valid,torch.ones_like(valid))
                scores=scores.masked_fill(~valid[:,:,None],-1e4)
            pooled=torch.einsum('btk,btd->bkd',scores.softmax(1),regions).flatten(1)
            views.append(torch.cat([F.gelu(self.global_proj(halves['global'][index])),pooled,F.gelu(self.mid_proj(halves['mid'][index])),F.gelu(self.std_proj(halves['dispersion'][index]))],-1))
        views=torch.stack(views,dim=1)
        score=self.reliability(views).squeeze(-1)
        if not self.cfg.get('learn_scale_gate',True):score=score*0
        drop=self.cfg.get('scale_dropout',0.)
        if self.training and drop:
            chosen=torch.randint(0,2,(len(views),),device=views.device)
            dropping=torch.rand(len(views),device=views.device)<drop
            score[torch.arange(len(views),device=views.device),chosen]=torch.where(dropping,torch.full_like(score[:,0],-1e4),score[torch.arange(len(views),device=views.device),chosen])
        weights=score.softmax(1)
        return (views*weights[:,:,None]).sum(1),views,weights
