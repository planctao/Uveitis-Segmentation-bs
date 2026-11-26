"""Actual CUDA complete-task evidence and independent trainable-gradient replay."""
import json
import numpy as np
import pandas as pd
import torch
from experiment import KEYS,normalize,loss_fn,sample_weights
from register_patch_value import RegisterPatchValueModel
from register_patch_query_features import BANK_KEYS,array_sha
from audit_register_patch_value import independent_probability
from audit_normal_tangent import complete_task

DETAILS=['p','subtype','selected_attention','baseline_pool','value_pool_difference','pool_difference','attention_difference','tail_difference']


from register_patch_query_witness import tensors_for,weights_for


def write_witness(path,model,source,features,bank,frame,fit,ids,config,seed):
    device=next(model.parameters()).device;base,selected,registers=tensors_for(features,bank,ids,source,device)
    y,disease,weights=weights_for(frame,fit,ids,source,device)
    out=model(base,selected,registers);task=loss_fn(out,y,disease,weights,source['cfg'])
    parameters=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
    gradients=torch.autograd.grad(task,[p for _,p in parameters])
    proof={'indices':np.asarray(ids),'fit_indices':np.asarray(fit),'seed':np.array(seed),'config':np.array(json.dumps(config,sort_keys=True)),
        'loss':np.array(float(task.detach())),'trainable_parameter_names':np.asarray([name for name,_ in parameters]),'gradient':torch.cat([g.flatten() for g in gradients]).detach().cpu().numpy()}
    for name,p in parameters:proof['initial_sha256_'+name]=np.array(array_sha(p.detach().cpu().numpy()))
    for key in DETAILS:proof['output_sha256_'+key]=np.array(array_sha(out[key].detach().cpu().numpy()))
    proof['probability']=out['p'].detach().cpu().numpy()
    for key in BANK_KEYS:proof['bank_sha256_'+key]=np.array(array_sha(np.asarray(bank[key][ids])))
    for key in features:proof['raw_sha256_'+key]=np.array(array_sha(features[key][ids]))
    np.savez_compressed(path,**proof)
    return task,out


def audit_witness(path,source,tail,features,bank,frame,fit,config,seed):
    with np.load(path) as archive:proof={key:archive[key] for key in archive.files}
    assert json.loads(str(proof['config']))==config and int(proof['seed'])==seed and np.array_equal(proof['fit_indices'],fit)
    ids=proof['indices'];expected=fit[np.random.default_rng(20291900+seed).permutation(len(fit))[:min(256,len(fit))]]
    assert np.array_equal(ids,expected)
    for key in BANK_KEYS:assert str(proof['bank_sha256_'+key])==array_sha(np.asarray(bank[key][ids]))
    for key in features:assert str(proof['raw_sha256_'+key])==array_sha(features[key][ids])
    device=torch.device('cuda');model=RegisterPatchValueModel(source,tail,config,20291900+seed).to(device).train()
    parameters={name:p for name,p in model.named_parameters() if p.requires_grad};assert list(parameters)==proof['trainable_parameter_names'].tolist()
    for name,p in parameters.items():assert str(proof['initial_sha256_'+name])==array_sha(p.detach().cpu().numpy())
    base,selected,registers=tensors_for(features,bank,ids,source,device);out=model(base,selected,registers)
    for key in DETAILS:assert str(proof['output_sha256_'+key])==array_sha(out[key].detach().cpu().numpy()),key
    raw={key:torch.from_numpy(features[key][ids]).to(device) for key in features}
    independent_state={name:p.detach().clone().requires_grad_() for name,p in parameters.items()}
    tail_state={name:value.detach().to(device) for name,value in tail.state_dict().items()}
    manual=independent_probability(source,tail_state,independent_state,raw,selected,config)
    assert torch.allclose(manual['p'],out['p'],atol=5e-5,rtol=5e-5) and torch.equal(manual['p'].argmax(1),out['p'].argmax(1))
    y,disease,weights=weights_for(frame,fit,ids,source,device);task=complete_task(manual,y,disease,weights,source['cfg'])
    assert abs(float(task.detach())-float(proof['loss']))<2e-5;task.backward()
    gradient=torch.cat([independent_state[name].grad.flatten() for name in parameters]).detach().cpu().numpy()
    error=float(np.abs(gradient-proof['gradient']).max());assert np.allclose(gradient,proof['gradient'],atol=1e-6,rtol=3e-3),error
    assert not any(p.requires_grad for p in model.head.parameters()) and not any(p.requires_grad for p in model.tail.parameters())
    return {'actual_images':len(ids),'trainable_parameter_names':list(parameters),'trainable_parameters':sum(p.numel() for p in parameters.values()),'complete_task_gradient_error':error,
        'original_cuda_outputs_sha256_restored':True,'frozen_feature_bank_inputs_sha256_restored':True,'frozen_head_tail_verified':True}
