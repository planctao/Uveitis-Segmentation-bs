"""Low-rank distributional regional summaries with an exact-identity readout."""
import torch
from torch import nn
import torch.nn.functional as F
from experiment import Model


def regional_quantiles(values,mask=None):
    batch,count,rank=values.shape
    valid=torch.ones((batch,count),dtype=torch.bool,device=values.device) if mask is None else mask>.5
    valid=torch.where(valid.any(1,keepdim=True),valid,torch.ones_like(valid))
    ordered=values.masked_fill(~valid[:,:,None],torch.inf).sort(dim=1).values
    position=(valid.sum(1)-1).to(values.dtype)[:,None]*values.new_tensor([.25,.5,.75])[None]
    lower=position.floor().long();upper=position.ceil().long();fraction=position-lower
    a=ordered.gather(1,lower[:,:,None].expand(-1,-1,rank))
    b=ordered.gather(1,upper[:,:,None].expand(-1,-1,rank))
    return a+(b-a)*fraction[:,:,None],valid


class QuantileSummary(nn.Module):
    def __init__(self,rank=16,kind='quantile'):
        super().__init__();assert rank in [16,32] and kind in ['mean','quantile']
        self.rank=rank;self.kind=kind;self.norm=nn.LayerNorm(128)
        self.down=nn.Linear(128,rank);self.readout=nn.Linear(3*rank,256,bias=False)
        nn.init.zeros_(self.readout.weight)

    def forward(self,tokens,mask=None):
        values=F.gelu(self.down(self.norm(tokens)))
        if self.kind=='quantile':summary,valid=regional_quantiles(values,mask)
        else:
            valid=torch.ones(values.shape[:2],dtype=torch.bool,device=values.device) if mask is None else mask>.5
            valid=torch.where(valid.any(1,keepdim=True),valid,torch.ones_like(valid))
            mean=(values*valid[:,:,None]).sum(1)/valid.sum(1)[:,None]
            summary=mean[:,None,:].expand(-1,3,-1)
        return .1*self.readout(summary.flatten(1)),summary,values,valid


class QuantileModel(Model):
    def __init__(self,cfg,dims,n_diseases):
        super().__init__(cfg,dims,n_diseases)
        assert cfg['arch']=='attention' and cfg['mode']=='flat' and cfg['taxonomy_only'] and cfg['hidden']==256
        with torch.random.fork_rng(devices=[]):
            self.quantile_summary=QuantileSummary(cfg['quantile_rank'],cfg['quantile_kind'])
        self.region_proj.register_forward_hook(self._capture)
        self.encoder[1].register_forward_hook(self._add)

    def _capture(self,module,args,output):
        self._quantile_input=F.gelu(output)

    def _add(self,module,args,output):
        self._quantile_delta,self._quantile_summary,self._quantile_values,self._quantile_valid=self.quantile_summary(self._quantile_input,self._quantile_mask)
        self._quantile_original=output
        return output if self.cfg.get('quantile_disabled',False) else output+self._quantile_delta

    def forward(self,x):
        self._quantile_mask=x.get('region_mask') if self.cfg.get('use_region_mask',True) else None
        out=super().forward(x)
        out.update(quantile_input=self._quantile_input,quantile_delta=self._quantile_delta,quantile_summary=self._quantile_summary,
                   quantile_values=self._quantile_values,quantile_valid=self._quantile_valid,quantile_original=self._quantile_original)
        return out
