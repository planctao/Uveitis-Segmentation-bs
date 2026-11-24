"""Appearance-conditioned affine residuals on original M5 hidden activations."""
import torch
from torch import nn
from experiment import Model


class AppearanceCalibration(nn.Module):
    def __init__(self,rank=8,conditioned=True):
        super().__init__();assert rank in [8,16]
        self.conditioned=conditioned
        self.down=nn.Sequential(nn.Linear(12,rank),nn.GELU())
        self.readout=nn.Linear(rank,512)
        nn.init.zeros_(self.readout.weight);nn.init.zeros_(self.readout.bias)

    def forward(self,style,hidden):
        context=style if self.conditioned else torch.zeros_like(style)
        scale,shift=self.readout(self.down(context)).chunk(2,dim=-1)
        gamma=.1*scale.tanh();beta=.1*shift
        return gamma*hidden+beta,gamma,beta


class AppearanceModel(Model):
    def __init__(self,cfg,dims,n_diseases):
        super().__init__(cfg,dims,n_diseases)
        assert cfg['arch']=='attention' and cfg['mode']=='flat' and cfg['taxonomy_only'] and cfg['hidden']==256 and dims['style']==12
        with torch.random.fork_rng(devices=[]):
            self.appearance=AppearanceCalibration(cfg['appearance_rank'],cfg['appearance_conditioned'])
        self.encoder[1].register_forward_hook(self._calibrate)

    def _calibrate(self,module,args,output):
        self._appearance_hidden=output
        self._appearance_delta,self._appearance_gamma,self._appearance_beta=self.appearance(self._appearance_style,output)
        return output if self.cfg.get('appearance_disabled',False) else output+self._appearance_delta

    def forward(self,x):
        self._appearance_style=x['style'];out=super().forward(x)
        out.update(appearance_input=self._appearance_style,appearance_hidden=self._appearance_hidden,
                   appearance_delta=self._appearance_delta,appearance_gamma=self._appearance_gamma,appearance_beta=self._appearance_beta)
        return out
