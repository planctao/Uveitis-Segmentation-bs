"""Independent geometry and complete first-training-task reconstruction."""
import hashlib
import numpy as np
import pandas as pd
import torch
from experiment import KEYS,Model,normalize,sample_weights
from normal_subspace import independent_transform
from verify_normal_subspace import independent_geometry as independent_normal_geometry
from audit_normal_tangent import complete_task,plain_replay


def independent_geometry(frame,normalized,geometry,count):
    diagnostic=independent_normal_geometry(frame,normalized,geometry)
    normal=frame.label.eq(2).to_numpy();names=sorted(frame.loc[normal,'patient_group'].unique())
    centers=[]
    for name in names:
        ids=np.flatnonzero(normal&frame.patient_group.eq(name).to_numpy())
        centers.append(normalized['global'][ids].astype('float64').sum(0)/len(ids))
    centers=np.stack(centers);deviation=np.sqrt(geometry['global_eigenvalues'])
    assert np.array_equal(deviation,geometry['risk_deviation'])
    points=((centers-geometry['global_center'])@geometry['global_basis'])/deviation
    assignment=np.asarray([np.argmin([np.square(p-c).sum() for c in geometry['risk_clusters']]) for p in points])
    assert np.allclose(points,geometry['risk_patient_points'],atol=2e-6,rtol=2e-6)
    assert np.array_equal(assignment,geometry['risk_patient_assignment'])
    for cluster in range(count):
        assert np.any(assignment==cluster)
        assert np.allclose(points[assignment==cluster].mean(0),geometry['risk_clusters'][cluster],atol=2e-6,rtol=2e-6)
    diagnostic['risk_clusters']={'normal_patients':len(names),'cluster_counts':np.bincount(assignment,minlength=count).tolist()}
    return diagnostic


def independent_penalty(probability,y,weight,groups,count,smoothing):
    logp=probability.clamp(min=1e-7).log();rows=[]
    for i in range(len(y)):rows.append(-(1-smoothing)*logp[i,y[i]]-smoothing*torch.sum(logp[i])/3)
    rows=torch.stack(rows);risks=[];masses=[];penalties=[]
    for parent in range(3):
        r=[];m=[];present=[]
        for group in range(count):
            ids=torch.nonzero((y==parent)&(groups==group),as_tuple=False).flatten()
            mass=weight[ids].sum()
            risk=(rows[ids]*weight[ids]).sum()/mass.clamp(min=1e-12)
            r.append(risk);m.append(mass)
            if len(ids):present.append(risk)
        risks.append(torch.stack(r));masses.append(torch.stack(m))
        if present:
            # Pairwise variance identity, independent of the vectorized var implementation.
            differences=torch.stack([a-b for a in present for b in present])
            penalties.append(differences.square().sum()/(2*len(present)**2))
    penalty=torch.stack(penalties).sum()/len(penalties) if penalties else rows.sum()*0
    return penalty,torch.stack(risks),torch.stack(masses)


def audit_training_witness(c,folder,features,frame,fit):
    device=torch.device('cuda',torch.cuda.current_device())
    with np.load(folder/'training_input_witness.npz') as bank:proof={k:bank[k] for k in bank.files}
    cfg=c['cfg'];n=int(proof['batch_size']);seed=cfg['seed'];assert n==min(len(fit),256)
    assert int(proof['seed'])==seed and int(proof['groups'])==cfg['normal_risk_groups'] and float(proof['strength'])==cfg['normal_risk_weight']
    generator=torch.Generator(device=device).manual_seed(20261124+seed)
    local=torch.randperm(len(fit),device=device,generator=generator)[:n].cpu().numpy();ids=fit[local]
    raw,_=normalize({k:v[ids] for k,v in features.items()},None,c['stats'])
    assert np.array_equal(raw['global'],proof['raw_global_all'])
    for key in KEYS:assert np.array_equal(raw[key][:8],proof['input_'+key])
    tf=frame.iloc[fit];mapping={v:i for i,v in enumerate(c['diseases'])}
    assert np.array_equal(proof['y'],frame.iloc[ids].label.to_numpy())
    assert np.array_equal(proof['disease'],frame.iloc[ids].disease.map(mapping).to_numpy())
    expected={'leaf':sample_weights(tf,tf.label)[local],'patient_ids':pd.factorize(tf.patient_group)[0].astype('int64')[local]}
    for name,value in expected.items():assert np.array_equal(value,proof['weight_'+name])
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(20261124+seed);model=Model(cfg,c['dims'],len(c['diseases']))
    for name,value in model.named_parameters():assert hashlib.sha256(value.detach().numpy().tobytes()).hexdigest()==str(proof['initial_parameter_sha256_'+name])
    model=model.to(device).train()
    data={k:torch.from_numpy(independent_transform(raw[k],c['state_dict']['normal_center_'+k].numpy(),c['state_dict']['normal_basis_'+k].numpy(),1.)).to(device) for k in KEYS}
    points=((raw['global'].astype('float64')-c['state_dict']['normal_center_global'].numpy())@c['state_dict']['normal_basis_global'].numpy())/c['state_dict']['risk_deviation'].numpy()
    groups=np.asarray([np.argmin([np.square(p-center).sum() for center in c['state_dict']['risk_clusters'].numpy()]) for p in points])
    assert np.allclose(points,proof['risk_points'],atol=2e-5,rtol=2e-5) and np.array_equal(groups,proof['risk_group'])
    out=plain_replay(model,data,proof);y=torch.from_numpy(proof['y']).to(device);disease=torch.from_numpy(proof['disease']).to(device)
    weights={k:torch.from_numpy(v).to(device) for k,v in expected.items()}
    base=complete_task(out,y,disease,weights,cfg)
    penalty,risks,masses=independent_penalty(out['p'],y,weights['leaf'],torch.from_numpy(groups).to(device),cfg['normal_risk_groups'],cfg['label_smoothing'])
    total=base+cfg['normal_risk_weight']*penalty
    assert np.allclose(out['p'].detach().cpu(),proof['probability'],atol=5e-5,rtol=5e-5)
    assert np.allclose(risks.detach().cpu(),proof['class_group_risks'],atol=2e-5,rtol=2e-5)
    assert np.allclose(masses.detach().cpu(),proof['class_group_masses'],atol=2e-5,rtol=2e-5)
    for actual,name in [(base,'base_loss'),(penalty,'penalty'),(total,'total_loss')]:assert abs(float(actual.detach())-float(proof[name]))<2e-5
    return {'batch_size':n,'full_task_restored':True,'risk_variance_restored':float(penalty.detach()),
        'class_group_batch_counts':[[int(np.sum((proof['y']==y)&(groups==g))) for g in range(cfg['normal_risk_groups'])] for y in range(3)]}
