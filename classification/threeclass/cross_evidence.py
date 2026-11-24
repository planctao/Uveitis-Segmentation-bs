"""Low-rank additive/product interactions between global and pooled local evidence."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from experiment import Model


class CrossEvidence(nn.Module):
    def __init__(self, rank=16, product=True):
        super().__init__();assert rank in [16,32]
        self.rank=rank;self.product=product
        self.global_norm=nn.LayerNorm(256);self.local_norm=nn.LayerNorm(256)
        self.global_down=nn.Linear(256,rank);self.local_down=nn.Linear(256,rank)
        self.pair_norm=nn.LayerNorm(rank)
        self.readout=nn.Linear(rank,256,bias=False);nn.init.zeros_(self.readout.weight)

    def forward(self, fused):
        assert fused.shape[-1]==640
        a=F.gelu(self.global_down(self.global_norm(fused[:,:256])))
        b=F.gelu(self.local_down(self.local_norm(fused[:,256:512])))
        pair=a*b if self.product else (a+b)/math.sqrt(2)
        return .1*self.readout(self.pair_norm(pair))


class CrossEvidenceModel(Model):
    def __init__(self,cfg,dims,n_diseases):
        super().__init__(cfg,dims,n_diseases)
        assert cfg['arch']=='attention' and cfg['mode']=='flat' and cfg['taxonomy_only'] and cfg['hidden']==256
        assert cfg.get('region_heads',2)==2
        with torch.random.fork_rng(devices=[]):
            self.cross_evidence=CrossEvidence(cfg['cross_evidence_rank'],cfg['cross_evidence_product'])
        self.encoder[0].register_forward_pre_hook(self._capture)
        self.encoder[1].register_forward_hook(self._add)

    def _capture(self,module,args):
        self._cross_fused=args[0]

    def _add(self,module,args,output):
        self._cross_delta=self.cross_evidence(self._cross_fused);self._cross_original_hidden=output
        return output if self.cfg.get('cross_evidence_disabled',False) else output+self._cross_delta

    def forward(self,x):
        out=super().forward(x)
        out.update(cross_evidence_input=self._cross_fused,cross_evidence_delta=self._cross_delta,cross_evidence_original_hidden=self._cross_original_hidden)
        return out
