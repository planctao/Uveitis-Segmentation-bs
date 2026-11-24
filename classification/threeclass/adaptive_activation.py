"""Bounded basis residuals on the original hidden GELU activation."""
import torch
from torch import nn
from experiment import Model


class ActivationResidual(nn.Module):
    def __init__(self,kind='spline',groups=16,channels=256):
        super().__init__();assert kind in ['spline','gaussian'] and groups in [16,256] and channels==256
        self.kind=kind;self.groups=groups
        self.register_buffer('centers',torch.linspace(-3,3,5))
        self.register_buffer('channel_group',torch.arange(channels)//(channels//groups))
        self.coefficients=nn.Parameter(torch.zeros(groups,5))

    def basis(self,x):
        bounded=3*torch.tanh(x/3)
        distance=(bounded[:,:,None]-self.centers)/1.5
        if self.kind=='spline':return (1-distance.abs()).clamp(min=0)
        return (-.5*distance.square()).softmax(-1)

    def forward(self,x):
        bases=self.basis(x)
        return .1*(bases*self.coefficients[self.channel_group]).sum(-1),bases


class ActivationModel(Model):
    def __init__(self,cfg,dims,n_diseases):
        super().__init__(cfg,dims,n_diseases)
        assert cfg['arch']=='attention' and cfg['mode']=='flat' and cfg['taxonomy_only'] and cfg['hidden']==256
        with torch.random.fork_rng(devices=[]):
            self.activation_residual=ActivationResidual(cfg['activation_kind'],cfg['activation_groups'])
        self.encoder[1].register_forward_hook(self._activation)

    def _activation(self,module,args,output):
        self._activation_input=args[0];self._activation_original=output
        self._activation_delta,self._activation_basis=self.activation_residual(args[0])
        return output if self.cfg.get('activation_disabled',False) else output+self._activation_delta

    def forward(self,x):
        out=super().forward(x)
        out.update(activation_input=self._activation_input,activation_delta=self._activation_delta,
                   activation_basis=self._activation_basis,activation_original=self._activation_original)
        return out
