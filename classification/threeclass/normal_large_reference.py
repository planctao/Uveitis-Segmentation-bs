"""Normal-reference DINOv3-L/combined features; unchanged original head and task."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from experiment import loss_fn
from normal_subspace import NormalSuppressedModel


class NormalLargeModel(NormalSuppressedModel):
    def __init__(self,cfg,dims,n_diseases,geometry=None,witness_path=None):
        super().__init__(cfg,dims,n_diseases,geometry)
        assert cfg['normal_rank']==32 and cfg['normal_strength']==1.
        self.witness_path=None if witness_path is None else Path(witness_path);self.witness_written=False

    def forward(self,x):
        context=None
        if self.training and torch.is_grad_enabled() and self.witness_path is not None and not self.witness_written:
            device=x['global'].device
            context={'model':self,'raw_global_all':x['global'],'rng_cpu':torch.random.get_rng_state().clone(),
                'rng_cuda':torch.cuda.get_rng_state(device).clone() if device.type=='cuda' else None,
                'initial_parameter_sha256':{k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in self.named_parameters()}}
        out=super().forward(x)
        if context is not None:
            projected=self.transform(x);context.update(projected_global=projected['global'],projected_region_mean=projected['regions'].mean(1))
            out['_large_normal_witness_context']=context
        return out


def normal_large_loss(out,y,disease,weights,cfg):
    loss=loss_fn(out,y,disease,weights,cfg);context=out.get('_large_normal_witness_context')
    if context is not None:
        model=context['model'];proof={'seed':np.array(cfg['seed']),'batch_size':np.array(len(y)),
            'feature':np.array(cfg['normal_large_feature']),'loss':np.array(float(loss.detach())),'rng_cpu':context['rng_cpu'].numpy()}
        tensors={'raw_global_all':context['raw_global_all'],'projected_global':context['projected_global'],'projected_region_mean':context['projected_region_mean'],
            'y':y,'disease':disease,'probability':out['p'],'subtype':out['subtype']}
        tensors.update({'weight_'+k:v for k,v in weights.items()})
        for k,v in tensors.items():proof[k]=v.detach().cpu().numpy()
        if context['rng_cuda'] is not None:proof['rng_cuda']=context['rng_cuda'].cpu().numpy()
        for k,v in context['initial_parameter_sha256'].items():proof['initial_parameter_sha256_'+k]=np.array(v)
        model.witness_path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(model.witness_path,**proof);model.witness_written=True
    return loss
