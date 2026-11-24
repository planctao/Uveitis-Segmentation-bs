"""Soft diversity constraint; attention may still overlap below a fixed margin."""
import torch
import torch.nn.functional as F


def attention_overlap_penalty(attention, margin):
    if attention.ndim!=3 or attention.shape[-1]<2 or not 0<=margin<1:
        raise ValueError('Expected B,T,H attention with H>=2 and margin in [0,1)')
    normalized=F.normalize(attention,dim=1)
    gram=torch.einsum('bth,btk->bhk',normalized,normalized)
    row,col=torch.triu_indices(attention.shape[-1],attention.shape[-1],offset=1,device=attention.device)
    return (gram[:,row,col]-margin).clamp(min=0).square().mean(1)
