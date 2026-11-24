"""Low-rank, symmetric scale interactions supplement the shared scale mean."""
import torch
from torch import nn
import torch.nn.functional as F
from scale_head import ScaleConsensusPool


class ScaleInteractionPool(ScaleConsensusPool):
    def __init__(self,dims,cfg):
        super().__init__(dims,cfg)
        self.base_dimension=self.dimension
        rank=cfg.get('interaction_rank',64)
        self.kind=cfg.get('interaction_kind','both')
        if self.kind not in ['difference','product','both']:
            raise ValueError('Interaction kind must be difference, product or both')
        self.pair_projection=nn.Sequential(nn.Linear(self.base_dimension,rank),nn.LayerNorm(rank),nn.GELU())
        self.interaction_dimension=rank*(2 if self.kind=='both' else 1)
        self.interaction_gate=nn.Parameter(torch.full((self.interaction_dimension,),-1.))
        self.dimension+=self.interaction_dimension

    def pair_features(self,views):
        a,b=self.pair_projection(views).unbind(1)
        parts=[]
        if self.kind in ['difference','both']:parts.append((a-b).abs())
        if self.kind in ['product','both']:parts.append(a*b)
        return torch.cat(parts,dim=-1)*self.interaction_gate.sigmoid()

    def forward(self,x):
        fused,views,weights=super().forward(x)
        pair=self.pair_features(views)
        drop=self.cfg.get('interaction_dropout',0.)
        if not 0<=drop<=1:raise ValueError('Interaction dropout must be in [0,1]')
        if self.training and drop:
            pair=pair*(torch.rand(len(pair),1,device=pair.device)>=drop)
        fused=torch.cat([fused,pair],dim=-1)
        # The shared auxiliary classifier sees each scale without pair evidence.
        view_padding=views.new_zeros(len(views),2,self.interaction_dimension)
        return fused,torch.cat([views,view_padding],dim=-1),weights
