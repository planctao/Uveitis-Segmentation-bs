"""Independent explicit patient aggregation and correlation matrices."""
import hashlib
import numpy as np
import pandas as pd
import torch
from experiment import KEYS,Model,normalize,sample_weights
from normal_subspace import independent_transform
from audit_normal_tangent import complete_task,plain_replay


def numpy_context(raw,state):
    parts=[]
    for key in KEYS:
        value=raw[key].astype('float64')
        if value.ndim==3:value=value.sum(1)/value.shape[1]
        coordinate=np.einsum('nd,dk->nk',value-state['normal_center_'+key].numpy().astype('float64'),state['normal_basis_'+key].numpy().astype('float64'))
        parts.append(np.clip(coordinate/state['decorrelation_deviation_'+key].numpy(),-5,5))
    return np.concatenate(parts,1).astype('float32')


def independent_penalty(hidden,context,y,disease,patient,scope):
    pairs=sorted(set(zip(patient.detach().cpu().tolist(),disease.detach().cpu().tolist())))
    means=[];coordinates=[];labels=[]
    for p,d in pairs:
        ids=torch.nonzero((patient==p)&(disease==d),as_tuple=False).flatten()
        means.append(torch.stack([hidden[i] for i in ids]).sum(0)/len(ids))
        coordinates.append(torch.stack([context[i] for i in ids]).sum(0)/len(ids))
        assert torch.unique(y[ids]).numel()==1
        labels.append(int(y[ids[0]]) if scope=='coarse' else d)
    means=torch.stack(means);coordinates=torch.stack(coordinates);terms=[];sizes=[]
    for label in sorted(set(labels)):
        ids=[i for i,v in enumerate(labels) if v==label]
        if len(ids)<4:continue
        a=means[ids];b=coordinates[ids];a=a-a.sum(0)/len(ids);b=b-b.sum(0)/len(ids)
        matrix=torch.einsum('ni,nj->ij',a,b)/len(ids)
        a2=(a*a).sum(0)/len(ids);b2=(b*b).sum(0)/len(ids)
        terms.append(((matrix*matrix)/(a2[:,None]*b2[None,:]+1e-4)).sum()/matrix.numel());sizes.append(len(ids))
    return (torch.stack(terms).sum()/len(terms) if terms else hidden.sum()*0),sizes


def numpy_penalty(hidden,context,y,disease,patient,scope):
    rows=[];coords=[];labels=[]
    for p,d in sorted(set(zip(patient.tolist(),disease.tolist()))):
        ids=np.flatnonzero((patient==p)&(disease==d));rows.append(hidden[ids].astype('float64').mean(0));coords.append(context[ids].astype('float64').mean(0));labels.append(int(y[ids[0]]) if scope=='coarse' else int(d))
    rows=np.stack(rows);coords=np.stack(coords);labels=np.array(labels);terms=[];sizes=[]
    for label in sorted(set(labels)):
        ids=np.flatnonzero(labels==label)
        if len(ids)<4:continue
        a=rows[ids]-rows[ids].mean(0);b=coords[ids]-coords[ids].mean(0);cross=a.T@b/len(ids)
        terms.append(np.mean(cross**2/(np.mean(a*a,0)[:,None]*np.mean(b*b,0)[None,:]+1e-4)));sizes.append(len(ids))
    return (float(np.mean(terms)) if terms else 0.),sizes


def independent_forward(plain,raw,state,proof=None):
    device=next(plain.parameters()).device;hidden=[]
    handle=plain.encoder[1].register_forward_hook(lambda module,args,output:hidden.append(output))
    data={k:torch.from_numpy(independent_transform(raw[k],state['normal_center_'+k].numpy(),state['normal_basis_'+k].numpy(),1.)).to(device) for k in KEYS}
    try:out=plain(data) if proof is None else plain_replay(plain,data,proof)
    finally:handle.remove()
    out.update(decorrelation_hidden=hidden[0],decorrelation_context=torch.from_numpy(numpy_context(raw,state)).to(device));return out


def audit_training_witness(c,folder,features,frame,fit):
    with np.load(folder/'training_input_witness.npz') as bank:proof={k:bank[k] for k in bank.files}
    cfg=c['cfg'];seed=cfg['seed'];n=int(proof['batch_size']);device=torch.device('cuda',torch.cuda.current_device())
    assert n==min(len(fit),256) and int(proof['seed'])==seed
    assert str(proof['scope'])==cfg['normal_decorrelation_scope'] and float(proof['strength'])==cfg['normal_decorrelation_strength']
    local=torch.randperm(len(fit),device=device,generator=torch.Generator(device=device).manual_seed(20261124+seed))[:n].cpu().numpy()
    ids=fit[local];raw,_=normalize({k:v[ids] for k,v in features.items()},None,c['stats']);assert np.array_equal(raw['global'],proof['raw_global_all'])
    tf=frame.iloc[fit];mapping={v:i for i,v in enumerate(c['diseases'])}
    assert np.array_equal(proof['y'],frame.iloc[ids].label.to_numpy()) and np.array_equal(proof['disease'],frame.iloc[ids].disease.map(mapping).to_numpy())
    weights={'leaf':sample_weights(tf,tf.label)[local],'patient_ids':pd.factorize(tf.patient_group)[0].astype('int64')[local]}
    for k,v in weights.items():assert np.array_equal(v,proof['weight_'+k])
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(20261124+seed);plain=Model(cfg,c['dims'],len(c['diseases']))
    for k,v in plain.named_parameters():assert hashlib.sha256(v.detach().numpy().tobytes()).hexdigest()==str(proof['initial_parameter_sha256_'+k])
    plain=plain.to(device).train();out=independent_forward(plain,raw,c['state_dict'],proof)
    for key in ['decorrelation_context','decorrelation_hidden']:
        assert np.allclose(out[key].detach().cpu().numpy(),proof[key],atol=5e-5,rtol=5e-5)
    assert np.allclose(out['p'].detach().cpu(),proof['probability'],atol=5e-5,rtol=5e-5)
    y=torch.from_numpy(proof['y']).to(device);d=torch.from_numpy(proof['disease']).to(device);wt={k:torch.from_numpy(v).to(device) for k,v in weights.items()}
    base=complete_task(out,y,d,wt,cfg);penalty,sizes=independent_penalty(out['decorrelation_hidden'],out['decorrelation_context'],y,d,wt['patient_ids'],cfg['normal_decorrelation_scope'])
    numpy_value,numpy_sizes=numpy_penalty(proof['decorrelation_hidden'],proof['decorrelation_context'],proof['y'],proof['disease'],proof['weight_patient_ids'],cfg['normal_decorrelation_scope'])
    assert sizes==numpy_sizes==proof['stratum_sizes'].tolist()
    assert abs(numpy_value-float(proof['penalty']))<2e-6 and abs(float(penalty.detach())-float(proof['penalty']))<2e-6
    total=base+cfg['normal_decorrelation_strength']*penalty
    assert abs(float(base.detach())-float(proof['base_loss']))<2e-5 and abs(float(total.detach())-float(proof['loss']))<2e-5
    return {'batch_size':n,'full_task_restored':True,'independent_patient_penalty_restored':True,'stratum_sizes':sizes,'numpy_penalty_error':abs(numpy_value-float(proof['penalty']))}
