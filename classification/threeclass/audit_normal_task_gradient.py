"""Independent auxiliary task gradients and asymmetric representation projection."""
import hashlib
import numpy as np
import pandas as pd
import torch
from experiment import KEYS,Model,normalize,sample_weights
from normal_subspace import independent_transform
from audit_normal_tangent import plain_replay,complete_task


def independent_coarse(out,y,weight,smoothing):
    logp=out['p'].clamp(min=1e-7).log()
    terms=[(-(1-smoothing)*logp[i,y[i]]-smoothing*logp[i].sum()/3)*weight[i] for i in range(len(y))]
    return torch.stack(terms).sum()/len(y)


def numpy_repair(coarse,auxiliary,mode,strength):
    a=coarse.astype('float64');b=auxiliary.astype('float64');correction=np.zeros_like(a)
    groups=[np.arange(len(a))] if mode=='batch' else [np.asarray([i]) for i in range(len(a))]
    for ids in groups:
        dot=np.sum(a[ids]*b[ids]);norm=max(float(np.sum(a[ids]**2)),1e-20)
        if dot<0:correction[ids]=-strength*dot/norm*a[ids]
    return correction.astype('float32')


def audit_training_witness(c,folder,features,frame,fit):
    device=torch.device('cuda',torch.cuda.current_device())
    with np.load(folder/'training_input_witness.npz') as bank:proof={k:bank[k] for k in bank.files}
    cfg=c['cfg'];seed=cfg['seed'];n=int(proof['batch_size']);assert n==min(len(fit),256)
    assert int(proof['seed'])==seed and str(proof['mode'])==cfg['normal_task_gradient_mode'] and float(proof['strength'])==cfg['normal_task_gradient_strength']
    generator=torch.Generator(device=device).manual_seed(20261124+seed)
    local=torch.randperm(len(fit),device=device,generator=generator)[:n].cpu().numpy();ids=fit[local]
    raw,_=normalize({k:v[ids] for k,v in features.items()},None,c['stats']);assert np.array_equal(raw['global'],proof['raw_global_all'])
    tf=frame.iloc[fit];mapping={v:i for i,v in enumerate(c['diseases'])}
    assert np.array_equal(proof['y'],frame.iloc[ids].label.to_numpy()) and np.array_equal(proof['disease'],frame.iloc[ids].disease.map(mapping).to_numpy())
    expected={'leaf':sample_weights(tf,tf.label)[local],'patient_ids':pd.factorize(tf.patient_group)[0].astype('int64')[local]}
    for key,value in expected.items():assert np.array_equal(value,proof['weight_'+key])
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(20261124+seed);model=Model(cfg,c['dims'],len(c['diseases']))
    for key,value in model.named_parameters():assert hashlib.sha256(value.detach().numpy().tobytes()).hexdigest()==str(proof['initial_parameter_sha256_'+key])
    model=model.to(device).train()
    data={k:torch.from_numpy(independent_transform(raw[k],c['state_dict']['normal_center_'+k].numpy(),c['state_dict']['normal_basis_'+k].numpy(),1.)).to(device) for k in KEYS}
    out=plain_replay(model,data,proof);y=torch.from_numpy(proof['y']).to(device);d=torch.from_numpy(proof['disease']).to(device)
    weights={k:torch.from_numpy(v).to(device) for k,v in expected.items()}
    base=complete_task(out,y,d,weights,cfg);coarse=independent_coarse(out,y,weights['leaf'],cfg['label_smoothing']);aux=base-coarse
    gc=torch.autograd.grad(coarse,out['embedding'],retain_graph=True)[0].cpu().numpy()
    ga=torch.autograd.grad(aux,out['embedding'])[0].cpu().numpy()
    assert np.allclose(gc,proof['coarse_gradient'],atol=2e-7,rtol=3e-3)
    assert np.allclose(ga,proof['auxiliary_gradient'],atol=2e-7,rtol=3e-3)
    assert np.allclose(out['p'].detach().cpu(),proof['probability'],atol=5e-5,rtol=5e-5)
    assert np.allclose(out['embedding'].detach().cpu(),proof['embedding'],atol=5e-5,rtol=5e-5)
    for actual,key in [(base,'base_loss'),(coarse,'coarse_loss'),(aux,'auxiliary_loss')]:assert abs(float(actual.detach())-float(proof[key]))<2e-5
    assert float(proof['base_loss'])==float(proof['total_loss'])
    delta=numpy_repair(proof['coarse_gradient'],proof['auxiliary_gradient'],str(proof['mode']),float(proof['strength']))
    assert np.allclose(delta,proof['gradient_correction'],atol=2e-8,rtol=2e-5)
    regenerated=numpy_repair(gc,ga,str(proof['mode']),float(proof['strength']))
    assert np.allclose(regenerated,proof['gradient_correction'],atol=2e-7,rtol=3e-3)
    axes=1 if str(proof['mode'])=='image' else (0,1)
    before=(proof['coarse_gradient']*proof['auxiliary_gradient']).sum(axes)
    after=(proof['coarse_gradient']*(proof['auxiliary_gradient']+delta)).sum(axes)
    expected_dot=before-float(proof['strength'])*np.minimum(before,0)
    assert np.allclose(after,expected_dot,atol=2e-10,rtol=1e-4)
    return {'batch_size':n,'full_task_restored':True,'task_gradient_repair_restored':True,
        'negative_dot_fraction':float(np.mean(before<0)),'maximum_independent_correction_error':float(np.max(np.abs(regenerated-proof['gradient_correction'])))}
