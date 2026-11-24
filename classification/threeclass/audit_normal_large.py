"""Independent initial full task, normal projection and feature-source witness."""
import hashlib
import numpy as np
import pandas as pd
import torch
from experiment import KEYS,Model,normalize,sample_weights
from normal_subspace import independent_transform
from audit_normal_tangent import complete_task,plain_replay


def audit_training_witness(c,folder,features,frame,fit):
    with np.load(folder/'training_input_witness.npz') as bank:proof={k:bank[k] for k in bank.files}
    cfg=c['cfg'];seed=cfg['seed'];n=int(proof['batch_size']);device=torch.device('cuda',torch.cuda.current_device())
    assert n==min(len(fit),256) and int(proof['seed'])==seed and str(proof['feature'])==cfg['normal_large_feature']
    local=torch.randperm(len(fit),device=device,generator=torch.Generator(device=device).manual_seed(20261124+seed))[:n].cpu().numpy()
    ids=fit[local];raw,_=normalize({k:v[ids] for k,v in features.items()},None,c['stats']);assert np.array_equal(raw['global'],proof['raw_global_all'])
    tf=frame.iloc[fit];mapping={v:i for i,v in enumerate(c['diseases'])}
    assert np.array_equal(proof['y'],frame.iloc[ids].label.to_numpy()) and np.array_equal(proof['disease'],frame.iloc[ids].disease.map(mapping).to_numpy())
    weights={'leaf':sample_weights(tf,tf.label)[local],'patient_ids':pd.factorize(tf.patient_group)[0].astype('int64')[local]}
    for k,v in weights.items():assert np.array_equal(v,proof['weight_'+k])
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(20261124+seed);plain=Model(cfg,c['dims'],len(c['diseases']))
    for k,v in plain.named_parameters():assert hashlib.sha256(v.detach().numpy().tobytes()).hexdigest()==str(proof['initial_parameter_sha256_'+k])
    projected={k:independent_transform(raw[k],c['state_dict']['normal_center_'+k].numpy(),c['state_dict']['normal_basis_'+k].numpy(),1.) for k in KEYS}
    maximum=max(float(np.max(np.abs(projected['global']-proof['projected_global']))),float(np.max(np.abs(projected['regions'].mean(1)-proof['projected_region_mean']))))
    assert np.allclose(projected['global'],proof['projected_global'],atol=6e-5,rtol=6e-5)
    assert np.allclose(projected['regions'].mean(1),proof['projected_region_mean'],atol=6e-5,rtol=6e-5)
    out=plain_replay(plain.to(device).train(),{k:torch.from_numpy(v).to(device) for k,v in projected.items()},proof)
    for key,stored in [('p','probability'),('subtype','subtype')]:assert np.allclose(out[key].detach().cpu(),proof[stored],atol=6e-5,rtol=6e-5)
    loss=complete_task(out,torch.from_numpy(proof['y']).to(device),torch.from_numpy(proof['disease']).to(device),{k:torch.from_numpy(v).to(device) for k,v in weights.items()},cfg)
    assert abs(float(loss.detach())-float(proof['loss']))<2e-5
    return {'batch_size':n,'feature':cfg['normal_large_feature'],'full_task_restored':True,'normal_projection_restored':True,'maximum_projection_error':maximum}
