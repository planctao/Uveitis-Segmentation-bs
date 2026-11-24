"""Independent functional spatial contrast path and actual first-task replay."""
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from experiment import KEYS,Model,normalize,sample_weights
from normal_subspace import independent_transform
from normal_spatial_contrast import NormalSpatialContrastModel
from audit_normal_tangent import complete_task,plain_replay


def manual_branch(tokens,state,cfg):
    centered=tokens-torch.sum(tokens,1,keepdim=True)/9
    z=F.layer_norm(centered,(128,),state['spatial_contrast.norm.weight'],state['spatial_contrast.norm.bias'],1e-5)
    values=F.gelu(F.linear(z,state['spatial_contrast.down.weight'],state['spatial_contrast.down.bias']))
    summaries=[]
    if cfg['spatial_contrast_kind']=='d4':
        grid=values.reshape(len(values),3,3,-1)
        views=[torch.rot90(grid.flip(2) if flip else grid,k,dims=(1,2)).flatten(1) for flip in [False,True] for k in range(4)]
    else:views=[(values.sum(1)/9).repeat(1,9)]
    for view in views:summaries.append(F.gelu(F.linear(view,state['spatial_contrast.mix.weight'],state['spatial_contrast.mix.bias'])))
    summary=sum(summaries)/len(summaries)
    delta=.1*F.linear(summary,state['spatial_contrast.readout.weight'],state['spatial_contrast.readout.bias'])
    return delta,centered,values,summary


def independent_forward(plain,raw,state,cfg,proof=None):
    device=next(plain.parameters()).device;details={};adapter={k:v.to(device) for k,v in state.items() if k.startswith('spatial_contrast.')}
    def capture(module,args,output):details['spatial_tokens']=F.gelu(output)
    def add(module,args,hidden):
        delta,centered,values,summary=manual_branch(details['spatial_tokens'],adapter,cfg)
        details.update(spatial_hidden=hidden,spatial_delta=delta,spatial_centered=centered,spatial_values=values,spatial_summary=summary)
        return hidden+delta
    projected={k:torch.from_numpy(independent_transform(raw[k],state['normal_center_'+k].detach().cpu().numpy(),state['normal_basis_'+k].detach().cpu().numpy(),1.)).to(device) for k in KEYS}
    handles=[plain.region_proj.register_forward_hook(capture),plain.encoder[1].register_forward_hook(add)]
    try:out=plain(projected) if proof is None else plain_replay(plain,projected,proof)
    finally:
        for handle in handles:handle.remove()
    out.update(details);return out


def audit_training_witness(c,folder,features,frame,fit):
    with np.load(folder/'training_input_witness.npz') as bank:proof={k:bank[k] for k in bank.files}
    cfg=c['cfg'];seed=cfg['seed'];n=int(proof['batch_size']);device=torch.device('cuda',torch.cuda.current_device())
    assert n==min(len(fit),256) and int(proof['seed'])==seed
    assert str(proof['kind'])==cfg['spatial_contrast_kind'] and int(proof['rank'])==cfg['spatial_contrast_rank']
    local=torch.randperm(len(fit),device=device,generator=torch.Generator(device=device).manual_seed(20261124+seed))[:n].cpu().numpy()
    ids=fit[local];raw,_=normalize({k:v[ids] for k,v in features.items()},None,c['stats']);assert np.array_equal(raw['global'],proof['raw_global_all'])
    tf=frame.iloc[fit];mapping={v:i for i,v in enumerate(c['diseases'])}
    assert np.array_equal(proof['y'],frame.iloc[ids].label.to_numpy()) and np.array_equal(proof['disease'],frame.iloc[ids].disease.map(mapping).to_numpy())
    weights={'leaf':sample_weights(tf,tf.label)[local],'patient_ids':pd.factorize(tf.patient_group)[0].astype('int64')[local]}
    for k,v in weights.items():assert np.array_equal(v,proof['weight_'+k])
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(20261124+seed);initial=NormalSpatialContrastModel(cfg,c['dims'],len(c['diseases']))
    state=initial.state_dict()
    for k,v in initial.named_parameters():assert hashlib.sha256(v.detach().numpy().tobytes()).hexdigest()==str(proof['initial_parameter_sha256_'+k])
    for k,v in c['state_dict'].items():
        if k.startswith('normal_'):state[k]=v
    plain=Model(cfg,c['dims'],len(c['diseases']));plain.load_state_dict({k:v for k,v in state.items() if not k.startswith(('normal_','spatial_contrast.'))},strict=True)
    out=independent_forward(plain.to(device).train(),raw,state,cfg,proof);maximum=0.
    for key in ['spatial_tokens','spatial_hidden','spatial_delta','spatial_centered','spatial_values','spatial_summary']:
        actual=out[key].detach().cpu().numpy();assert np.allclose(actual,proof[key],atol=6e-5,rtol=6e-5),key
        maximum=max(maximum,float(np.max(np.abs(actual-proof[key]))))
    for key,stored in [('p','probability'),('subtype','subtype')]:assert np.allclose(out[key].detach().cpu(),proof[stored],atol=6e-5,rtol=6e-5)
    assert np.max(np.abs(proof['spatial_delta']))==0
    loss=complete_task(out,torch.from_numpy(proof['y']).to(device),torch.from_numpy(proof['disease']).to(device),{k:torch.from_numpy(v).to(device) for k,v in weights.items()},cfg)
    assert abs(float(loss.detach())-float(proof['loss']))<2e-5
    return {'batch_size':n,'full_task_restored':True,'centered_spatial_contrast_restored':True,'zero_initial_delta':True,'maximum_independent_forward_error':maximum}
