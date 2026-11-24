"""Protect coarse-task representation gradients from opposing auxiliary gradients."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from experiment import loss_fn
from normal_subspace import NormalSuppressedModel


def coarse_task(out,y,weights,cfg):
    logp=out['p'].clamp(min=1e-7).log();smooth=cfg.get('label_smoothing',0.)
    rows=-(1-smooth)*logp.gather(1,y[:,None]).squeeze(1)-smooth*logp.mean(1)
    return (rows*weights['leaf']).mean()


def repair_gradient(coarse,auxiliary,mode,strength):
    assert mode in ['image','batch'] and strength in [0.,.5,1.]
    axes=1 if mode=='image' else (0,1)
    dot=(coarse*auxiliary).sum(axes,keepdim=True)
    denominator=coarse.square().sum(axes,keepdim=True).clamp(min=1e-20)
    delta=-strength*(dot.clamp(max=0)/denominator)*coarse
    return delta


class NormalTaskGradientModel(NormalSuppressedModel):
    def __init__(self,cfg,dims,n_diseases,geometry=None,witness_path=None):
        super().__init__(cfg,dims,n_diseases,geometry)
        assert cfg['normal_task_gradient_mode'] in ['image','batch'] and cfg['normal_task_gradient_strength'] in [0.,.5,1.]
        self.witness_path=None if witness_path is None else Path(witness_path);self.witness_written=False

    def forward(self,x):
        context=None
        if self.training and torch.is_grad_enabled() and self.witness_path is not None and not self.witness_written:
            device=x['global'].device
            context={'model':self,'raw_global_all':x['global'],'rng_cpu':torch.random.get_rng_state().clone(),
                'rng_cuda':torch.cuda.get_rng_state(device).clone() if device.type=='cuda' else None,
                'initial_parameter_sha256':{k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in self.named_parameters()}}
        out=super().forward(x)
        if context is not None:out['_task_gradient_context']=context
        return out


def task_gradient_loss(out,y,disease,weights,cfg):
    base=loss_fn(out,y,disease,weights,cfg);coarse=coarse_task(out,y,weights,cfg);auxiliary=base-coarse
    gc=torch.autograd.grad(coarse,out['embedding'],retain_graph=True)[0]
    ga=torch.autograd.grad(auxiliary,out['embedding'],retain_graph=True)[0]
    delta=repair_gradient(gc,ga,cfg['normal_task_gradient_mode'],cfg['normal_task_gradient_strength']).detach()
    # Value is exactly zero. Its gradient changes only ancestors of the shared embedding.
    total=base+((out['embedding']-out['embedding'].detach())*delta).sum()
    context=out.get('_task_gradient_context')
    if context is not None:
        model=context['model'];proof={'seed':np.asarray(cfg['seed']),'mode':np.asarray(cfg['normal_task_gradient_mode']),
            'strength':np.asarray(cfg['normal_task_gradient_strength']),'batch_size':np.asarray(len(y)),
            'base_loss':np.asarray(float(base.detach())),'coarse_loss':np.asarray(float(coarse.detach())),
            'auxiliary_loss':np.asarray(float(auxiliary.detach())),'total_loss':np.asarray(float(total.detach())),
            'rng_cpu':context['rng_cpu'].numpy()}
        tensors={'raw_global_all':context['raw_global_all'],'y':y,'disease':disease,'probability':out['p'],
            'embedding':out['embedding'],'coarse_gradient':gc,'auxiliary_gradient':ga,'gradient_correction':delta}
        tensors.update({'weight_'+k:v for k,v in weights.items()})
        for key,value in tensors.items():proof[key]=value.detach().cpu().numpy()
        if context['rng_cuda'] is not None:proof['rng_cuda']=context['rng_cuda'].cpu().numpy()
        for key,value in context['initial_parameter_sha256'].items():proof['initial_parameter_sha256_'+key]=np.asarray(value)
        model.witness_path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(model.witness_path,**proof);model.witness_written=True
    out['_task_gradient_details']={'coarse':gc,'auxiliary':ga,'correction':delta}
    return total
