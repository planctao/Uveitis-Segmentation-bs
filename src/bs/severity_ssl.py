"""Mask/clinical-label-free DEC and SwAV numerical mechanisms."""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F


def student_assignments(latent,centers):
    distance=torch.cdist(latent,centers).square()
    weights=(1+distance).reciprocal()
    return weights/weights.sum(1,keepdim=True)


@torch.no_grad()
def dec_target(q):
    weights=q.square()/q.sum(0,keepdim=True).clamp_min(1e-12)
    return weights/weights.sum(1,keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def sinkhorn_assignments(logits,epsilon=.05,iterations=3):
    """Single-device author-equivalent Sinkhorn, computed in log space."""
    logq=(logits/epsilon).T
    k,b=logq.shape
    logq=logq-torch.logsumexp(logq.reshape(-1),0)
    for _ in range(iterations):
        logq=logq-torch.logsumexp(logq,dim=1,keepdim=True)-math.log(k)
        logq=logq-torch.logsumexp(logq,dim=0,keepdim=True)-math.log(b)
    return (logq+math.log(b)).exp().T


def swapped_prediction_loss(outputs,targets,temperature=.1):
    losses=[]
    for crop_id,q in enumerate(targets):
        other=[-(q*F.log_softmax(output/temperature,dim=1)).sum(1).mean()
               for view,output in enumerate(outputs) if view!=crop_id]
        losses.append(torch.stack(other).mean())
    return torch.stack(losses).mean()


class DECEncoder(nn.Module):
    def __init__(self,input_dim=768):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(input_dim,500),nn.ReLU(),nn.Linear(500,500),nn.ReLU(),
                                   nn.Linear(500,2000),nn.ReLU(),nn.Linear(2000,10))
        self.decoder=nn.Sequential(nn.Linear(10,2000),nn.ReLU(),nn.Linear(2000,500),nn.ReLU(),
                                   nn.Linear(500,500),nn.ReLU(),nn.Linear(500,input_dim))

    def forward(self,x):
        return self.decoder(self.encoder(x))


class SwAVProjection(nn.Module):
    def __init__(self,input_dim=768):
        super().__init__()
        self.projection=nn.Sequential(nn.Linear(input_dim,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Linear(512,128))
        self.prototypes=nn.Linear(128,4,bias=False)

    def forward(self,features):
        z=F.normalize(self.projection(features),dim=1)
        return z,self.prototypes(z)

    @torch.no_grad()
    def normalize_prototypes(self):
        self.prototypes.weight.copy_(F.normalize(self.prototypes.weight,dim=1))
