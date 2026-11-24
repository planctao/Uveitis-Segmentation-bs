"""Train-only stochastic removal of at most one original M5 evidence group."""
from pathlib import Path
import numpy as np
import torch
from experiment import Model

LENGTHS=(256,256,64,64)


def draw_gains(count,probability,rng):
    assert 0<=probability<=1
    chosen=rng.integers(0,4,size=count)
    dropping=rng.random(count)<probability
    gains=np.full((count,4),1/(1-probability/4),dtype='float32')
    gains[np.flatnonzero(dropping),chosen[dropping]]=0
    return gains,chosen,dropping


def apply_gains(fused,gains):
    return fused*torch.repeat_interleave(gains,torch.tensor(LENGTHS,device=fused.device),dim=1)


class EvidenceDropoutModel(Model):
    def __init__(self,cfg,dims,n_diseases,witness_path=None):
        super().__init__(cfg,dims,n_diseases)
        assert cfg['arch']=='attention' and cfg['mode']=='flat' and cfg['taxonomy_only'] and cfg['hidden']==256
        assert cfg.get('region_heads',2)==2
        self.evidence_rng=np.random.default_rng(20281124+cfg.get('seed',0))
        self.witness_path=None if witness_path is None else Path(witness_path);self.witness_written=False
        self.encoder.register_forward_pre_hook(self._drop)

    def _drop(self,module,args):
        probability=self.cfg['evidence_drop_probability']
        if not self.training or not probability:return None
        fused=args[0];gains,chosen,dropping=draw_gains(len(fused),probability,self.evidence_rng)
        output=apply_gains(fused,torch.as_tensor(gains,device=fused.device,dtype=fused.dtype))
        if self.witness_path is not None and not self.witness_written:
            self.witness_path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(self.witness_path,source=fused.detach().cpu().numpy(),output=output.detach().cpu().numpy(),
                gains=gains,chosen=chosen,dropping=dropping,probability=probability,seed=self.cfg['seed'])
            self.witness_written=True
        return (output,)
