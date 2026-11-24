"""Independent spatial moments/ridge solution and complete training-task replay."""
import hashlib
import numpy as np
import pandas as pd
import torch
from experiment import KEYS,Model,normalize,sample_weights
from normal_subspace import independent_transform
from normal_spatial_reference import independent_spatial_transform
from verify_normal_subspace import independent_geometry as check_normal_geometry
from audit_normal_tangent import complete_task,plain_replay


def independent_geometry(frame,normalized,geometry):
    diagnostics=check_normal_geometry(frame,normalized,geometry)
    normal=frame.label.eq(2).to_numpy();names=sorted(frame.loc[normal,'patient_group'].unique());targets=[];globals=[]
    basis=geometry['regions_basis'].astype('float64');center=geometry['regions_center'].astype('float64')
    for name in names:
        ids=np.flatnonzero(normal&frame.patient_group.eq(name).to_numpy());spatial=[]
        for index in ids:
            raw=normalized['regions'][index].astype('float64');r=np.clip(raw-((raw-center)@basis)@basis.T,-10,10).astype('float32').astype('float64')
            spatial.append(r-r.sum(0,keepdims=True)/r.shape[0])
        targets.append(np.stack(spatial).sum(0)/len(ids));globals.append(normalized['global'][ids].astype('float64').sum(0)/len(ids))
    target=np.stack(targets);x=np.stack(globals);dev=np.maximum(np.sqrt(geometry['global_eigenvalues'][:8]),.05)
    q=np.clip(np.einsum('nd,dk->nk',x-geometry['global_center'],geometry['global_basis'][:,:8].astype('float64'))/dev,-5,5)
    qm=q.sum(0)/len(q);mean=target.sum(0)/len(target);a=q-qm
    augmented=np.vstack([a,np.sqrt(len(a))*np.eye(8)]);response=np.vstack([(target-mean).reshape(len(a),-1),np.zeros((8,target.shape[1]*target.shape[2]))])
    coefficient=np.linalg.lstsq(augmented,response,rcond=None)[0].reshape(8,*target.shape[1:])
    assert np.allclose(dev,geometry['spatial_deviation'],atol=1e-6)
    assert np.allclose(qm,geometry['spatial_qmean'],atol=1e-6)
    assert np.allclose(mean,geometry['spatial_center'],atol=1e-6)
    assert np.allclose(coefficient,geometry['spatial_coefficients'],atol=1e-6,rtol=2e-5)
    assert np.allclose(geometry['spatial_center'].mean(0),0,atol=1e-6)
    assert np.allclose(geometry['spatial_coefficients'].mean(1),0,atol=1e-6)
    diagnostics['spatial']={'normal_patients':len(names),'ridge':1.,'context_rank':8,'maximum_independent_coefficient_error':float(np.max(np.abs(coefficient-geometry['spatial_coefficients'])))}
    return diagnostics


def checkpoint_geometry(c):
    state=c['state_dict'];g={}
    for key in KEYS:
        g[key+'_center']=state['normal_center_'+key].numpy();g[key+'_basis']=state['normal_basis_'+key].numpy()
    for key in ['spatial_qmean','spatial_deviation','spatial_center','spatial_coefficients']:g[key]=state[key].numpy()
    return g


def audit_training_witness(c,folder,features,frame,fit):
    with np.load(folder/'training_input_witness.npz') as bank:proof={k:bank[k] for k in bank.files}
    cfg=c['cfg'];seed=cfg['seed'];n=int(proof['batch_size']);device=torch.device('cuda',torch.cuda.current_device())
    assert n==min(len(fit),256) and int(proof['seed'])==seed
    assert bool(proof['conditioned'])==cfg['normal_spatial_conditioned'] and float(proof['strength'])==cfg['normal_spatial_strength']
    local=torch.randperm(len(fit),device=device,generator=torch.Generator(device=device).manual_seed(20261124+seed))[:n].cpu().numpy()
    ids=fit[local];raw,_=normalize({k:v[ids] for k,v in features.items()},None,c['stats']);assert np.array_equal(raw['global'],proof['raw_global_all'])
    tf=frame.iloc[fit];mapping={v:i for i,v in enumerate(c['diseases'])}
    assert np.array_equal(proof['y'],frame.iloc[ids].label.to_numpy()) and np.array_equal(proof['disease'],frame.iloc[ids].disease.map(mapping).to_numpy())
    weights={'leaf':sample_weights(tf,tf.label)[local],'patient_ids':pd.factorize(tf.patient_group)[0].astype('int64')[local]}
    for k,v in weights.items():assert np.array_equal(v,proof['weight_'+k])
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(20261124+seed);plain=Model(cfg,c['dims'],len(c['diseases']))
    for k,v in plain.named_parameters():assert hashlib.sha256(v.detach().numpy().tobytes()).hexdigest()==str(proof['initial_parameter_sha256_'+k])
    transformed=independent_spatial_transform(raw,checkpoint_geometry(c),cfg['normal_spatial_conditioned'],cfg['normal_spatial_strength'])
    assert np.allclose(transformed['regions'],proof['transformed_regions'],atol=5e-5,rtol=5e-5)
    plain=plain.to(device).train();out=plain_replay(plain,{k:torch.from_numpy(v).to(device) for k,v in transformed.items()},proof)
    assert np.allclose(out['p'].detach().cpu(),proof['probability'],atol=5e-5,rtol=5e-5)
    loss=complete_task(out,torch.from_numpy(proof['y']).to(device),torch.from_numpy(proof['disease']).to(device),{k:torch.from_numpy(v).to(device) for k,v in weights.items()},cfg)
    assert abs(float(loss.detach())-float(proof['loss']))<2e-5
    return {'batch_size':n,'full_task_restored':True,'spatial_input_restored':True,'maximum_input_error':float(np.max(np.abs(transformed['regions']-proof['transformed_regions'])))}
