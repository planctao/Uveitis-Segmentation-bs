"""Independent initial-batch reconstruction of complete normal-tangent objectives."""
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from experiment import KEYS, Model, normalize, sample_weights
from normal_subspace import independent_transform


def complete_task(out, y, disease, weights, cfg):
    assert cfg['mode']=='flat' and cfg['taxonomy_only'] and not cfg.get('focal_gamma',0)
    logp=out['p'].clamp(min=1e-7).log();smooth=cfg.get('label_smoothing',0.)
    coarse=-(1-smooth)*logp[torch.arange(len(y),device=y.device),y]-smooth*logp.mean(1)
    fine=out['subtype']-torch.logsumexp(out['subtype'],1,keepdim=True)
    loss=((coarse-cfg.get('aux',.6)*fine[torch.arange(len(y),device=y.device),disease])*weights['leaf']).mean()
    if cfg.get('fine_contrastive',0):
        emb=out['fine_embedding'];emb=emb/emb.norm(dim=1,keepdim=True).clamp(min=1e-12)
        scores=emb@emb.T/.15;scores=scores.masked_fill(torch.eye(len(y),device=y.device,dtype=torch.bool),-1e4)
        positive=(disease[:,None]==disease[None,:])&(weights['patient_ids'][:,None]!=weights['patient_ids'][None,:])
        counts=positive.sum(1);valid=counts>0
        logprob=scores-torch.logsumexp(scores,1,keepdim=True)
        if valid.any():loss=loss+cfg['fine_contrastive']*(-(logprob*positive).sum(1)/counts.clamp(min=1)*weights['leaf'])[valid].mean()
    return loss


def plain_replay(model, data, proof):
    device=data['global'].device;devices=[device.index] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.random.set_rng_state(torch.from_numpy(proof['rng_cpu'].copy()))
        if device.type=='cuda':torch.cuda.set_rng_state(torch.from_numpy(proof['rng_cuda'].copy()),device)
        return model(data)


def audit_training_witness(c, folder, features, frame, fit):
    device=torch.device('cuda',torch.cuda.current_device())
    with np.load(folder/'training_input_witness.npz') as saved:proof={k:saved[k] for k in saved.files}
    cfg=c['cfg'];n=int(proof['batch_size']);seed=cfg['seed'];eps=cfg['normal_tangent_epsilon']
    assert str(proof['mode'])==cfg['normal_tangent_mode'] and float(proof['epsilon'])==eps and int(proof['seed'])==seed
    assert n==min(len(fit),256)
    generator=torch.Generator(device=device).manual_seed(20261124+seed)
    local=torch.randperm(len(fit),device=device,generator=generator)[:n].cpu().numpy()
    ids=fit[local];raw,_=normalize({k:v[ids] for k,v in features.items()},None,c['stats'])
    assert np.array_equal(raw['global'],proof['raw_global_all'])
    for key in KEYS:assert np.array_equal(raw[key][:8],proof['input_'+key])
    tf=frame.iloc[fit];disease_index={v:i for i,v in enumerate(c['diseases'])}
    assert np.array_equal(proof['y'],frame.iloc[ids].label.to_numpy())
    assert np.array_equal(proof['disease'],frame.iloc[ids].disease.map(disease_index).to_numpy())
    expected_weights={'leaf':sample_weights(tf,tf.label)[local],'patient_ids':pd.factorize(tf.patient_group)[0].astype('int64')[local]}
    for key,value in expected_weights.items():assert np.array_equal(value,proof['weight_'+key])
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(20261124+seed)
        plain=Model(cfg,c['dims'],len(c['diseases']))
    for key,value in plain.named_parameters():
        assert hashlib.sha256(value.detach().numpy().tobytes()).hexdigest()==str(proof['initial_parameter_sha256_'+key])
    plain=plain.to(device).train()
    bases={};deviations={};base={};coordinates={}
    for key in KEYS:
        bases[key]=c['state_dict']['normal_basis_'+key].to(device)
        deviations[key]=c['state_dict']['tangent_deviation_'+key].numpy()
        values=independent_transform(raw[key],c['state_dict']['normal_center_'+key].numpy(),bases[key].cpu().numpy(),1.)
        assert np.allclose(values[:8],proof['base_'+key],atol=5e-5,rtol=5e-5)
        base[key]=torch.from_numpy(values).to(device)
        coordinates[key]=torch.zeros(n,32,device=device,requires_grad=True)
    def perturb(coords):
        result={}
        for key in KEYS:
            delta=coords[key]@bases[key].T
            if base[key].ndim==3:delta=delta[:,None]
            result[key]=(base[key]+delta).clamp(-10,10)
        return result
    y=torch.from_numpy(proof['y']).to(device);disease=torch.from_numpy(proof['disease']).to(device)
    weights={key:torch.from_numpy(value).to(device) for key,value in expected_weights.items()}
    out=plain_replay(plain,perturb(coordinates),proof);clean=complete_task(out,y,disease,weights,cfg)
    assert np.allclose(out['p'].detach().cpu().numpy(),proof['clean_probability'],atol=5e-5,rtol=5e-5)
    assert abs(float(clean.detach())-float(proof['clean_loss']))<1e-5
    gradients=torch.autograd.grad(clean,[coordinates[k] for k in KEYS]);rng=np.random.default_rng(20282124+seed)
    maximum_error=0.;minimum_sign_fraction=1.;attack={}
    for key,gradient in zip(KEYS,gradients):
        rebuilt=gradient.cpu().numpy();stored=proof['coordinate_gradient_'+key]
        assert np.allclose(rebuilt,stored,atol=2e-7,rtol=3e-3)
        maximum_error=max(maximum_error,float(np.max(np.abs(rebuilt-stored))))
        random=rng.choice(np.array([-1.,1.],dtype='float32'),(n,32))
        direction=np.sign(stored) if cfg['normal_tangent_mode']=='fgsm' else random
        assert np.array_equal(direction,proof['direction_'+key])
        strong=np.abs(rebuilt)>2e-7
        if cfg['normal_tangent_mode']=='fgsm' and strong.any():
            fraction=float(np.mean(np.sign(rebuilt[strong])==direction[strong]));assert fraction==1.
            minimum_sign_fraction=min(minimum_sign_fraction,fraction)
        coordinate=(eps*deviations[key]*direction).astype('float32')
        assert np.array_equal(coordinate,proof['coordinate_'+key])
        assert np.all(np.abs(coordinate)<=eps*deviations[key]+1e-7)
        attack[key]=torch.from_numpy(coordinate).to(device)
    perturbed=perturb(attack)
    for key in KEYS:assert np.allclose(perturbed[key][:8].cpu().numpy(),proof['output_'+key],atol=5e-5,rtol=5e-5)
    out=plain_replay(plain,perturbed,proof);adversarial=complete_task(out,y,disease,weights,cfg)
    assert np.allclose(out['p'].detach().cpu().numpy(),proof['perturbed_probability'],atol=5e-5,rtol=5e-5)
    assert abs(float(adversarial.detach())-float(proof['perturbed_loss']))<1e-5
    return {'batch_size':n,'maximum_independent_coordinate_gradient_error':maximum_error,
            'fgsm_nontrivial_gradient_sign_match_fraction':minimum_sign_fraction,'full_objective_restored':True}
