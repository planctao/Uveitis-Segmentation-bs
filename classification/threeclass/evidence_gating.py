"""Bounded input-dependent gains for four original DINO evidence streams."""
import torch
from torch import nn
import torch.nn.functional as F
from experiment import Model

LENGTHS=(256,256,64,64)


class EvidenceGains(nn.Module):
    def __init__(self,rank=8,conditioned=True):
        super().__init__();assert rank in [8,32]
        self.conditioned=conditioned
        self.norm=nn.LayerNorm(640)
        self.down=nn.Linear(640,rank)
        self.readout=nn.Linear(rank,4)
        nn.init.zeros_(self.readout.weight);nn.init.zeros_(self.readout.bias)

    def forward(self,fused):
        context=fused if self.conditioned else torch.zeros_like(fused)
        gains=1+.5*torch.tanh(self.readout(F.gelu(self.down(self.norm(context)))))
        expanded=torch.repeat_interleave(gains,torch.tensor(LENGTHS,device=gains.device),dim=-1)
        return fused*expanded,gains


class EvidenceGatingModel(Model):
    def __init__(self,cfg,dims,n_diseases):
        super().__init__(cfg,dims,n_diseases)
        assert cfg['arch']=='attention' and cfg['mode']=='flat' and cfg['taxonomy_only'] and cfg['hidden']==256
        assert cfg.get('region_heads',2)==2
        with torch.random.fork_rng(devices=[]):
            self.evidence_gains=EvidenceGains(cfg['evidence_gain_rank'],cfg['evidence_gain_conditioned'])
        self.encoder.register_forward_pre_hook(self._gate)

    def _gate(self,module,args):
        self._gain_input=args[0];self._gain_output,self._gains=self.evidence_gains(args[0])
        return (args[0] if self.cfg.get('evidence_gains_disabled',False) else self._gain_output,)

    def forward(self,x):
        out=super().forward(x)
        out.update(evidence_gain_input=self._gain_input,evidence_gain_output=self._gain_output,evidence_gains=self._gains)
        return out
