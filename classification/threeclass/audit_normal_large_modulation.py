"""Independent reconstruction of normal-B-query/L-region attention predictions and CUDA first tasks."""
import hashlib
import numpy as np
import pandas as pd
import torch
from experiment import KEYS,Model,normalize,sample_weights
from normal_large_modulation import NormalLargeModulationModel,independent_forward,DETAILS
from audit_normal_tangent import complete_task


def ordinary(checkpoint):
    dims={**checkpoint['dims'],**{k:1536 for k in KEYS}}
    model=Model(checkpoint['cfg'],dims,len(checkpoint['diseases']))
    model.load_state_dict({k:v for k,v in checkpoint['state_dict'].items()
        if not k.startswith(('normal_','large_normal_','large_residual.'))},strict=True)
    return model


@torch.inference_mode()
def independent_probability(checkpoint,features,indices):
    plain=ordinary(checkpoint).eval();probabilities=[]
    for start in range(0,len(indices),256):
        ids=indices[start:start+256];data,_=normalize({k:v[ids] for k,v in features.items()},None,checkpoint['stats'])
        out=independent_forward(plain,data,checkpoint['state_dict'],checkpoint['cfg']);probabilities.append(out['p'].numpy())
    return np.concatenate(probabilities)


def audit_training_witness(c,folder,features,frame,fit):
    with np.load(folder/'training_input_witness.npz') as saved:proof={k:saved[k] for k in saved.files}
    cfg=c['cfg'];seed=cfg['seed'];n=int(proof['batch_size']);device=torch.device('cuda',torch.cuda.current_device())
    assert n==min(256,len(fit)) and int(proof['seed'])==seed
    assert int(proof['rank'])==cfg['large_modulation_rank'] and str(proof['kind'])==cfg['large_modulation_kind']
    local=torch.randperm(len(fit),device=device,generator=torch.Generator(device=device).manual_seed(20261124+seed))[:n].cpu().numpy()
    ids=fit[local];raw,_=normalize({k:v[ids] for k,v in features.items()},None,c['stats'])
    assert np.array_equal(raw['global'],proof['raw_global_all'])
    tf=frame.iloc[fit];mapping={name:i for i,name in enumerate(c['diseases'])}
    assert np.array_equal(frame.iloc[ids].label.to_numpy(),proof['y'])
    assert np.array_equal(frame.iloc[ids].disease.map(mapping).to_numpy(),proof['disease'])
    weights={'leaf':sample_weights(tf,tf.label)[local],'patient_ids':pd.factorize(tf.patient_group)[0].astype('int64')[local]}
    for key,value in weights.items():assert np.array_equal(value,proof['weight_'+key])
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(20261124+seed);initial=NormalLargeModulationModel(cfg,c['dims'],len(c['diseases']))
    for name,p in initial.named_parameters():assert hashlib.sha256(p.detach().numpy().tobytes()).hexdigest()==str(proof['initial_parameter_sha256_'+name])
    initial=initial.to(device)
    state=initial.state_dict(keep_vars=True)
    for name,value in c['state_dict'].items():
        if name.startswith(('normal_','large_normal_')):state[name]=value
    plain=ordinary({**c,'state_dict':state}).to(device).train()
    with torch.random.fork_rng(devices=[device.index]):
        torch.random.set_rng_state(torch.from_numpy(proof['rng_cpu'].copy()))
        torch.cuda.set_rng_state(torch.from_numpy(proof['rng_cuda'].copy()),device)
        out=independent_forward(plain,raw,state,cfg)
    maximum=0.
    for key in DETAILS:
        value=out[key].detach().cpu().numpy();assert np.allclose(value,proof[key],atol=6e-5,rtol=6e-5),key
        maximum=max(maximum,float(np.abs(value-proof[key]).max()))
    assert np.count_nonzero(proof['large_residual_delta'])==0
    for key,saved in [('p','probability'),('subtype','subtype')]:assert np.allclose(out[key].detach().cpu().numpy(),proof[saved],atol=6e-5,rtol=6e-5)
    loss=complete_task(out,torch.from_numpy(proof['y']).to(device),torch.from_numpy(proof['disease']).to(device),{k:torch.from_numpy(v).to(device) for k,v in weights.items()},cfg)
    assert abs(float(loss.detach())-float(proof['loss']))<2e-5
    loss.backward()
    gradients=torch.cat([(p.grad if name.startswith('large_residual.') else dict(plain.named_parameters())[name].grad).flatten() for name,p in initial.named_parameters()]).cpu().numpy()
    error=float(np.max(np.abs(gradients-proof['parameter_gradient'])))
    assert np.allclose(gradients,proof['parameter_gradient'],atol=1e-6,rtol=3e-3),error
    return {'maximum_parameter_gradient_error':error,'batch_size':n,'full_task_restored':True,'initial_parameter_hashes_restored':True,'zero_initial_delta':True,
        'both_normal_spaces_and_large_hidden_modulation_restored':True,'maximum_independent_forward_error':maximum}
