"""Coarse-family morphology dictionaries with support-shrunk disease priors."""
import torch
from torch import nn
import torch.nn.functional as F


class SharedMorphology(nn.Module):
    def __init__(self,hidden,parents,counts,components,shrink=10.,init_std=.005):
        super().__init__()
        if set(parents)!={0,1,2} or len(parents)!=len(counts) or any(count<=0 for count in counts) or components<1 or shrink<0:
            raise ValueError('Three clinical families, positive fitting support and components required')
        self.components=components
        self.weight=nn.Parameter(torch.randn(3,components,hidden)*init_std)
        self.bias=nn.Parameter(torch.zeros(3,components))
        self.prior_logits=nn.Parameter(torch.zeros(len(parents),components))
        self.register_buffer('parents',torch.tensor(parents,dtype=torch.long))
        count=torch.tensor(counts,dtype=torch.float32)
        self.register_buffer('support_scale',(count/(count+shrink)).sqrt())

    def component_parameters(self):
        return self.weight-self.weight.mean(1,keepdim=True),self.bias-self.bias.mean(1,keepdim=True)

    def log_priors(self):
        return F.log_softmax(self.prior_logits*self.support_scale[:,None],dim=-1)

    def forward(self,embedding):
        weight,bias=self.component_parameters()
        family_components=torch.einsum('bh,ckh->bck',embedding,weight)+bias[None]
        components=family_components[:,self.parents]
        log_priors=self.log_priors()
        evidence=torch.logsumexp(components+log_priors[None],dim=-1)
        return evidence,components,log_priors

    def regularization(self):
        weight,bias=self.component_parameters()
        prior=self.prior_logits-self.prior_logits.mean(1,keepdim=True)
        return (weight.square().sum(-1)+bias.square()).mean()+prior.square().mean()
