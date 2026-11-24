"""Real development-batch initialization, gradients and L-side connectivity checks."""
import numpy as np
import pandas as pd
import torch
from experiment import ROOT,KEYS,Model,normalize,loss_fn,sample_weights,dump
from normal_subspace import NormalSuppressedModel
from normal_large_residual import NormalLargeResidualModel,fit_separate_geometry,independent_forward
from verify_normal_subspace import independent_geometry
from audit_normal_tangent import complete_task
from dinov3_large_data import load_derived
from kernel_residual_experiments import sha


def main():
    torch.set_num_threads(4)
    frame=pd.read_csv(ROOT/'metadata/manifest.csv');digest=sha(ROOT/'metadata/manifest.csv')
    features=load_derived(ROOT,frame,digest,'base_dual_large_dual')
    fit=np.flatnonzero(frame.split.eq('train'));mask=np.zeros(len(frame),bool);mask[fit]=True
    normalized,stats=normalize(features,mask);tf=frame.iloc[fit].reset_index(drop=True)
    data={k:v[fit] for k,v in normalized.items()};geometry=fit_separate_geometry(tf,data)
    diagnostics={}
    for branch,section,prefix in [('base',slice(None,1536),''),('large',slice(1536,None),'large_')]:
        g={k[len(prefix):]:v for k,v in geometry.items() if k.startswith(prefix)} if prefix else {k:v for k,v in geometry.items() if not k.startswith('large_')}
        diagnostics[branch]=independent_geometry(tf,{k:v[...,section] for k,v in data.items()},g)
    source=torch.load(ROOT/'runs/normal_subspace/normal_rank32_full/validation/seed_0/model.pt',map_location='cpu',weights_only=False)
    dims={**source['dims'],**{k:3584 for k in KEYS}}
    for k in KEYS:
        assert np.allclose(source['state_dict']['normal_center_'+k],geometry[k+'_center'],atol=1e-6,rtol=1e-6)
        assert np.allclose(source['state_dict']['normal_basis_'+k],geometry[k+'_basis'],atol=1e-6,rtol=1e-6)
    ids=np.random.default_rng(20284101).choice(len(tf),256,replace=False)
    raw={k:v[ids] for k,v in data.items()};x={k:torch.from_numpy(v) for k,v in raw.items()};base_x={k:v[...,:1536] for k,v in x.items()}
    y=torch.tensor(tf.iloc[ids].label.to_numpy());d=torch.tensor(tf.iloc[ids].disease.map({v:i for i,v in enumerate(source['diseases'])}).to_numpy())
    weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[ids]),'patient_ids':torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[ids])}
    cases=[]
    for rank in [16,32]:
        counts=[]
        for scope in ['global','all']:
            cfg={**source['cfg'],'large_residual_scope':scope,'large_residual_rank':rank}
            torch.manual_seed(20284102);base=NormalSuppressedModel(cfg,source['dims'],len(source['diseases']),geometry);rng=torch.random.get_rng_state().clone()
            torch.manual_seed(20284102);model=NormalLargeResidualModel(cfg,dims,len(source['diseases']),geometry)
            assert torch.equal(rng,torch.random.get_rng_state())
            for name,value in base.state_dict().items():assert torch.equal(value,model.state_dict()[name])
            base.train();model.train();rng=torch.random.get_rng_state().clone();out=model(x);actual=loss_fn(out,y,d,weights,cfg);actual.backward();after=torch.random.get_rng_state().clone()
            torch.random.set_rng_state(rng);original=loss_fn(base(base_x),y,d,weights,cfg);original.backward()
            assert torch.equal(actual,original) and torch.equal(after,torch.random.get_rng_state()) and torch.count_nonzero(out['large_residual_delta'])==0
            for name,p in base.named_parameters():assert torch.equal(p.grad,dict(model.named_parameters())[name].grad),name
            assert model.large_residual.readout.weight.grad.abs().sum()>0
            state=model.state_dict();state.update(source['state_dict']);model.load_state_dict(state,strict=True)
            with torch.no_grad():
                w=model.large_residual.readout.weight;w.copy_(torch.sin(torch.arange(w.numel()).reshape(w.shape)*.17)*.01)
                model.large_residual.readout.bias.copy_(torch.linspace(-.01,.01,256))
            model.zero_grad();plain=Model(cfg,source['dims'],len(source['diseases'])).train()
            state=model.state_dict(keep_vars=True)
            plain.load_state_dict({k:v for k,v in state.items() if not k.startswith(('normal_','large_normal_','large_residual.'))},strict=True)
            independent={k:v.detach().clone().requires_grad_(k.startswith('large_residual.')) for k,v in state.items()}
            rng=torch.random.get_rng_state().clone();out=model(x);actual=loss_fn(out,y,d,weights,cfg);actual.backward();after=torch.random.get_rng_state().clone()
            torch.random.set_rng_state(rng);manual_out=independent_forward(plain,raw,independent,cfg);manual=complete_task(manual_out,y,d,weights,cfg);manual.backward()
            assert torch.equal(after,torch.random.get_rng_state()) and torch.allclose(actual,manual,atol=6e-6,rtol=6e-5)
            errors=[]
            for name,p in model.named_parameters():
                other=independent[name].grad if name.startswith('large_residual.') else dict(plain.named_parameters())[name].grad
                assert p.grad is not None and other is not None and torch.allclose(p.grad,other,atol=1e-5,rtol=2e-3),name
                errors.append(float((p.grad-other).abs().max()))
            for name in ['p','large_residual_input','large_residual_hidden','large_residual_delta']:
                assert torch.allclose(out[name],manual_out[name],atol=7e-6,rtol=1e-4),name
            model.eval()
            with torch.no_grad():
                expected=model(x)['p'];changed={k:v.clone() for k,v in x.items()}
                for k in KEYS:changed[k][...,1536:]=changed[k].roll(1,0)[...,1536:]
                side_error=float((model(changed)['p']-expected).abs().max());assert side_error>1e-6
                altered={**x,'mid':x['mid'].clone()};altered['mid'][...,1536:]*=-1
                mid_error=float((model(altered)['p']-expected).abs().max())
                if scope=='global':assert mid_error==0
                else:assert mid_error>1e-6
            frozen={k:v.clone() for k,v in model.named_buffers()};before=model.large_residual.down.weight.detach().clone()
            torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.1).step();assert not torch.equal(before,model.large_residual.down.weight)
            for name,value in frozen.items():assert torch.equal(value,dict(model.named_buffers())[name])
            restored=NormalLargeResidualModel(cfg,dims,len(source['diseases'])).eval();restored.load_state_dict(model.state_dict(),strict=True)
            with torch.no_grad():assert torch.equal(model(x)['p'],restored(x)['p'])
            counts.append(sum(p.numel() for p in model.parameters()));cases.append({'rank':rank,'scope':scope,'batch_size':256,'parameters':counts[-1],
                'maximum_complete_task_gradient_error':max(errors),'large_side_probability_change':side_error,'large_mid_probability_change':mid_error})
        assert counts[0]==counts[1]
    dump(ROOT/'verification_normal_large_residual_draft.json',{'passed':True,'manifest_sha256':digest,'cases':cases,'normal_geometries':diagnostics,
        'code_sha256':{name:sha(ROOT/name) for name in ['normal_large_residual.py','verify_normal_large_residual.py']},
        'status':'CPU preflight only; no formal protocol or training fits; actual CUDA witness audit required before launch',
        'external_test_evaluated':False})
    print('Four real256image complete-task/all-parameter gradient checks, zero-init identity/RNG, L-side connectivity, matched capacity and strict restores passed.',flush=True)


if __name__=='__main__':main()
