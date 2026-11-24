"""Patient-support-shrunk mixtures of linear evidence within each disease."""
import math
import torch
from torch import nn


class SubclusterEvidence(nn.Module):
    def __init__(self, hidden, counts, components, shrink=10., init_std=.005):
        super().__init__()
        if components<1 or shrink<0 or any(count<=0 for count in counts):
            raise ValueError('Positive fitting support and component count required')
        self.components=components
        self.weight=nn.Parameter(torch.randn(len(counts),components,hidden)*init_std)
        self.bias=nn.Parameter(torch.zeros(len(counts),components))
        counts=torch.tensor(counts,dtype=torch.float32)
        self.register_buffer('support_scale',(counts/(counts+shrink)).sqrt())

    def centered_parameters(self):
        weight=self.weight-self.weight.mean(1,keepdim=True)
        bias=self.bias-self.bias.mean(1,keepdim=True)
        return weight*self.support_scale[:,None,None],bias*self.support_scale[:,None]

    def forward(self,embedding):
        weight,bias=self.centered_parameters()
        components=torch.einsum('bh,dkh->bdk',embedding,weight)+bias[None]
        evidence=torch.logsumexp(components,dim=-1)-math.log(self.components)
        return evidence,components

    def regularization(self):
        # Penalize the unscaled centered weights to retain shrinkage for rare diseases.
        weight=self.weight-self.weight.mean(1,keepdim=True)
        bias=self.bias-self.bias.mean(1,keepdim=True)
        return (weight.square().sum(-1)+bias.square()).mean()
