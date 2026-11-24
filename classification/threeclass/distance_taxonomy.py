"""Matched Euclidean/Poincare disease prototypes over original M5 hidden features."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from experiment import Model


def bounded_tangent(x):
    norm=x.norm(dim=-1,keepdim=True)
    return x*(3/norm.clamp(min=3))


def expmap_zero(x):
    norm=x.norm(dim=-1,keepdim=True);squared=norm.square()
    ratio=torch.where(norm<1e-4,1-squared/3+2*squared.square()/15,
        norm.tanh()/norm.clamp(min=1e-8))
    return ratio*x


def poincare_distance_squared(x,y):
    # Stable at coincident points; near zero the smooth squared-distance series applies.
    delta=(x[:,None]-y[None,:]).square().sum(-1)
    denominator=(1-x.square().sum(-1))[:,None]*(1-y.square().sum(-1))[None,:]
    ratio=delta/denominator.clamp(min=1e-8)
    regular=4*torch.asinh(ratio.clamp(min=1e-8).sqrt()).square()
    series=4*(ratio-ratio.square()/3+8*ratio.pow(3)/45)
    return torch.where(ratio<1e-4,series,regular)


class DistanceTaxonomy(nn.Module):
    def __init__(self,dimension,n_diseases,rank=32,geometry='euclidean'):
        super().__init__();assert rank in [16,32] and geometry in ['euclidean','poincare']
        self.geometry=geometry
        self.norm=nn.LayerNorm(dimension)
        self.project=nn.Linear(dimension,rank)
        self.prototypes=nn.Parameter(torch.randn(n_diseases,rank)*.05)
        self.log_scale=nn.Parameter(torch.tensor(math.log(10.)))
        self.bias=nn.Parameter(torch.zeros(n_diseases))

    def components(self,hidden):
        tangent=bounded_tangent(.1*self.project(self.norm(hidden)))
        prototypes=bounded_tangent(self.prototypes)
        if self.geometry=='euclidean':
            distance=4*(tangent[:,None]-prototypes[None,:]).square().sum(-1)
            embedding,centers=tangent,prototypes
        else:
            embedding,centers=expmap_zero(tangent),expmap_zero(prototypes)
            distance=poincare_distance_squared(embedding,centers)
        scale=self.log_scale.exp().clamp(min=1,max=100)
        return -scale*distance+self.bias,embedding,centers,distance,scale

    def forward(self,hidden):return self.components(hidden)[0]


class DistanceTaxonomyModel(Model):
    def __init__(self,cfg,dims,n_diseases):
        super().__init__(cfg,dims,n_diseases)
        assert cfg['arch']=='attention' and cfg['mode']=='flat' and cfg['taxonomy_only'] and cfg['hidden']==256
        with torch.random.fork_rng(devices=[]):
            self.flat=DistanceTaxonomy(256,n_diseases,cfg['distance_rank'],cfg['distance_geometry'])

    def forward(self,x):
        out=super().forward(x)
        # Original fine projection/hidden features and coarse marginalization remain intact.
        return out
