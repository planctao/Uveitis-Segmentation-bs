"""Independent normal spatial residual bank and paired-view full-task replay."""
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from experiment import KEYS,Model,normalize,sample_weights
from audit_normal_spatial import independent_geometry as check_spatial_geometry
from normal_subspace import independent_transform
from audit_normal_tangent import complete_task,plain_replay


def independent_geometry(frame,normalized,geometry):
    details=check_spatial_geometry(frame,normalized,geometry);residuals=[]
    for name in geometry['normal_patients']:
        ids=np.flatnonzero(frame.label.eq(2).to_numpy()&frame.patient_group.eq(name).to_numpy())
        x=normalized['global'][ids].astype('float64').mean(0)
        region=independent_transform(normalized['regions'][ids],geometry['regions_center'],geometry['regions_basis'],1.).astype('float64')
        target=(region-region.mean(1,keepdims=True)).mean(0)
        q=np.clip(((x-geometry['global_center'])@geometry['global_basis'][:,:8])/geometry['spatial_deviation'],-5,5)
        mean=geometry['spatial_center'].astype('float64').copy()
        for j in range(8):mean+=(q[j]-geometry['spatial_qmean'][j])*geometry['spatial_coefficients'][j]
        residuals.append(target-mean)
    bank=np.stack(residuals);bank-=bank.sum(0,keepdims=True)/len(bank);bank-=bank.sum(1,keepdims=True)/bank.shape[1]
    assert np.allclose(bank,geometry['consistency_bank'],atol=2e-6,rtol=2e-5)
    assert np.allclose(geometry['consistency_bank'].mean(0),0,atol=1e-6) and np.allclose(geometry['consistency_bank'].mean(1),0,atol=1e-6)
    details['bank']={'normal_patients':len(bank),'maximum_independent_bank_error':float(np.max(np.abs(bank-geometry['consistency_bank']))),'mean_square':float(np.mean(bank**2))}
    return details


def independent_js(a,b,weight):
    mean=(a+b)*.5
    terms=[]
    for index in range(len(a)):
        v=.5*(torch.sum(a[index]*(a[index].clamp(min=1e-7).log()-mean[index].clamp(min=1e-7).log()))+torch.sum(b[index]*(b[index].clamp(min=1e-7).log()-mean[index].clamp(min=1e-7).log())))
        terms.append(v*weight[index])
    return torch.stack(terms).sum()/len(a)


def numpy_js(a,b,weight):
    a=a.astype('float64');b=b.astype('float64');m=(a+b)/2
    return float(np.mean(.5*(np.sum(a*np.log(np.maximum(a,1e-7)/np.maximum(m,1e-7)),1)+np.sum(b*np.log(np.maximum(b,1e-7)/np.maximum(m,1e-7)),1))*weight))


def audit_training_witness(c,folder,features,frame,fit):
    with np.load(folder/'training_input_witness.npz') as bank:proof={k:bank[k] for k in bank.files}
    cfg=c['cfg'];seed=cfg['seed'];n=int(proof['batch_size']);device=torch.device('cuda',torch.cuda.current_device());assert n==min(len(fit),256) and int(proof['seed'])==seed
    assert str(proof['level'])==cfg['normal_consistency_level'] and float(proof['strength'])==cfg['normal_consistency_weight']
    local=torch.randperm(len(fit),device=device,generator=torch.Generator(device=device).manual_seed(20261124+seed))[:n].cpu().numpy();ids=fit[local]
    raw,_=normalize({k:v[ids] for k,v in features.items()},None,c['stats']);assert np.array_equal(raw['global'],proof['raw_global_all'])
    tf=frame.iloc[fit];mapping={v:i for i,v in enumerate(c['diseases'])}
    assert np.array_equal(proof['y'],frame.iloc[ids].label.to_numpy()) and np.array_equal(proof['disease'],frame.iloc[ids].disease.map(mapping).to_numpy())
    weights={'leaf':sample_weights(tf,tf.label)[local],'patient_ids':pd.factorize(tf.patient_group)[0].astype('int64')[local]}
    for k,v in weights.items():assert np.array_equal(v,proof['weight_'+k])
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(20261124+seed);plain=Model(cfg,c['dims'],len(c['diseases']))
    for k,v in plain.named_parameters():assert hashlib.sha256(v.detach().numpy().tobytes()).hexdigest()==str(proof['initial_parameter_sha256_'+k])
    projected={k:independent_transform(raw[k],c['state_dict']['normal_center_'+k].numpy(),c['state_dict']['normal_basis_'+k].numpy(),1.) for k in KEYS}
    bank=c['state_dict']['consistency_bank'].numpy();rng=np.random.default_rng(20283224+seed)
    first=rng.integers(0,len(bank),n);second=(first+rng.integers(1,len(bank),n))%len(bank)
    assert np.array_equal(first,proof['first']) and np.array_equal(second,proof['second']) and (first!=second).all()
    delta=(.5*(bank[first]-bank[second])/(2.**.5)).astype('float32');augmented=np.clip(projected['regions']+delta,-10,10).astype('float32')
    assert np.allclose(augmented,proof['augmented_regions'],atol=5e-5,rtol=5e-5)
    plain=plain.to(device).train();clean=plain_replay(plain,{k:torch.from_numpy(v).to(device) for k,v in projected.items()},proof)
    view=plain_replay(plain,{k:torch.from_numpy(augmented if k=='regions' else v).to(device) for k,v in projected.items()},proof)
    for actual,stored in [(clean['p'],'probability'),(clean['subtype'],'subtype'),(view['p'],'view_probability'),(view['subtype'],'view_subtype')]:assert np.allclose(actual.detach().cpu(),proof[stored],atol=5e-5,rtol=5e-5),stored
    base=complete_task(clean,torch.from_numpy(proof['y']).to(device),torch.from_numpy(proof['disease']).to(device),{k:torch.from_numpy(v).to(device) for k,v in weights.items()},cfg)
    a,b=(clean['p'],view['p']) if cfg['normal_consistency_level']=='coarse' else (F.softmax(clean['subtype'],-1),F.softmax(view['subtype'],-1))
    penalty=independent_js(a,b,torch.from_numpy(weights['leaf']).to(device));total=base+cfg['normal_consistency_weight']*penalty
    for actual,key in [(base,'base_loss'),(penalty,'penalty'),(total,'loss')]:assert abs(float(actual.detach())-float(proof[key]))<2e-5
    nv=numpy_js(a.detach().cpu().numpy(),b.detach().cpu().numpy(),weights['leaf']);assert abs(nv-float(proof['penalty']))<2e-6
    return {'batch_size':n,'full_task_restored':True,'donors_and_spatial_view_restored':True,'paired_dropout_and_penalty_restored':True,'numpy_js_error':abs(nv-float(proof['penalty']))}
