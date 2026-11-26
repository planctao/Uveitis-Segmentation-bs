"""Actual cached-image task witnesses and independent complete gradient reconstruction."""
import json
import numpy as np
import torch
from experiment import loss_fn
from audit_normal_tangent import complete_task
from register_patch_query_witness import tensors_for,weights_for
from register_patch_query_features import BANK_KEYS,array_sha
from class_query_patch_evidence import ClassQueryPatchEvidenceModel,TRACE
from audit_class_query_patch_evidence import independent_probability

DETAILS=['p','subtype']+TRACE


def write_witness(path,model,source,features,bank,frame,fit,ids,config,seed):
    device=next(model.parameters()).device
    base,selected,registers=tensors_for(features,bank,ids,source,device)
    y,disease,weights=weights_for(frame,fit,ids,source,device)
    out=model(base,selected,registers);task=loss_fn(out,y,disease,weights,source['cfg'])
    parameters=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    grads=torch.autograd.grad(task,[p for _,p in parameters])
    proof={'indices':np.asarray(ids),'fit_indices':np.asarray(fit),'seed':np.array(seed),
        'config':np.array(json.dumps(config,sort_keys=True)),'loss':np.array(float(task.detach())),
        'trainable_parameter_names':np.asarray([n for n,_ in parameters]),
        'gradient':torch.cat([g.flatten() for g in grads]).detach().cpu().numpy(),
        'probability':out['p'].detach().cpu().numpy()}
    for n,p in parameters:proof['initial_sha256_'+n]=np.array(array_sha(p.detach().cpu().numpy()))
    for key in DETAILS:proof['output_sha256_'+key]=np.array(array_sha(out[key].detach().cpu().numpy()))
    for key in BANK_KEYS:proof['bank_sha256_'+key]=np.array(array_sha(np.asarray(bank[key][ids])))
    for key in features:proof['raw_sha256_'+key]=np.array(array_sha(features[key][ids]))
    np.savez_compressed(path,**proof)


def audit_witness(path,source,tail,features,bank,frame,fit,config,seed):
    with np.load(path) as archive:proof={k:archive[k] for k in archive.files}
    assert json.loads(str(proof['config']))==config and int(proof['seed'])==seed
    assert np.array_equal(fit,proof['fit_indices'])
    ids=fit[np.random.default_rng(20293500+seed).permutation(len(fit))[:256]]
    assert np.array_equal(ids,proof['indices'])
    for key in BANK_KEYS:assert str(proof['bank_sha256_'+key])==array_sha(np.asarray(bank[key][ids]))
    for key in features:assert str(proof['raw_sha256_'+key])==array_sha(features[key][ids])
    model=ClassQueryPatchEvidenceModel(source,tail,config,20293500+seed).cuda().train()
    parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
    assert list(parameters)==proof['trainable_parameter_names'].tolist()
    for name,p in parameters.items():assert str(proof['initial_sha256_'+name])==array_sha(p.detach().cpu().numpy())
    base,selected,registers=tensors_for(features,bank,ids,source,'cuda')
    out=model(base,selected,registers)
    for key in DETAILS:assert str(proof['output_sha256_'+key])==array_sha(out[key].detach().cpu().numpy()),key
    raw={key:torch.from_numpy(value[ids]).cuda() for key,value in features.items()}
    state={n:p.detach().clone().requires_grad_() for n,p in parameters.items()}
    manual=independent_probability(source,{},state,raw,selected,config)
    assert torch.allclose(manual['p'],out['p'],atol=5e-5,rtol=5e-5)
    assert torch.equal(manual['p'].argmax(1),out['p'].argmax(1))
    y,disease,weights=weights_for(frame,fit,ids,source,'cuda')
    task=complete_task(manual,y,disease,weights,source['cfg'])
    assert abs(float(task.detach())-float(proof['loss']))<2e-5
    gradient=torch.cat([g.flatten() for g in torch.autograd.grad(task,[state[n] for n in parameters])]).cpu().numpy()
    error=float(np.abs(gradient-proof['gradient']).max())
    assert np.allclose(gradient,proof['gradient'],atol=1e-6,rtol=3e-3),error
    assert not any(p.requires_grad for p in model.head.parameters())
    assert not any(p.requires_grad for p in model.tail.parameters())
    return {'actual_images':len(ids),'trainable_parameter_names':list(parameters),
        'trainable_parameters':sum(p.numel() for p in parameters.values()),'complete_task_gradient_error':error,
        'original_cuda_outputs_sha256_restored':True,'frozen_feature_bank_inputs_sha256_restored':True,
        'frozen_head_tail_verified':True,'source_normal_geometry_and_all_original_parameters_frozen':True}
