"""Four full-task/all-parameter gradient cases and actual CUDA witness before fitting."""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment import ROOT,KEYS,Model,normalize,loss_fn,sample_weights,dump
from normal_subspace import NormalSuppressedModel
from normal_large_residual import fit_separate_geometry
from normal_large_cross_attention import NormalLargeCrossAttentionModel,independent_forward,cross_attention_loss
from audit_normal_large_cross_attention import audit_training_witness
from verify_normal_subspace import independent_geometry
from audit_normal_tangent import complete_task
from dinov3_large_data import load_derived
from kernel_residual_experiments import sha


def main():
    torch.set_num_threads(4);frame=pd.read_csv(ROOT/'metadata/manifest.csv');digest=sha(ROOT/'metadata/manifest.csv')
    features=load_derived(ROOT,frame,digest,'base_dual_large_dual');fit=np.flatnonzero(frame.split.eq('train'))
    mask=np.zeros(len(frame),bool);mask[fit]=True;normalized,stats=normalize(features,mask);tf=frame.iloc[fit].reset_index(drop=True)
    data={k:v[fit] for k,v in normalized.items()};geometry=fit_separate_geometry(tf,data);diagnostics={}
    for branch,section,prefix in [('base',slice(None,1536),''),('large',slice(1536,None),'large_')]:
        g={k[len(prefix):]:v for k,v in geometry.items() if k.startswith(prefix)} if prefix else {k:v for k,v in geometry.items() if not k.startswith('large_')}
        diagnostics[branch]=independent_geometry(tf,{k:v[...,section] for k,v in data.items()},g)
    source=torch.load(ROOT/'runs/normal_subspace/normal_rank32_full/validation/seed_0/model.pt',map_location='cpu',weights_only=False)
    dims={**source['dims'],**{k:3584 for k in KEYS}};n_diseases=len(source['diseases'])
    ids=np.random.default_rng(20284201).choice(len(tf),256,replace=False);raw={k:v[ids] for k,v in data.items()}
    x={k:torch.from_numpy(v) for k,v in raw.items()};bx={k:v[...,:1536] for k,v in x.items()}
    y=torch.tensor(tf.iloc[ids].label.to_numpy());d=torch.tensor(tf.iloc[ids].disease.map({v:i for i,v in enumerate(source['diseases'])}).to_numpy())
    weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[ids]),'patient_ids':torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[ids])};cases=[]
    for rank in [16,32]:
        counts=[]
        for kind in ['mean','attention']:
            cfg={**source['cfg'],'large_residual_scope':'all','large_residual_rank':32,'large_cross_rank':rank,'large_cross_kind':kind}
            torch.manual_seed(20284202);base=NormalSuppressedModel(cfg,source['dims'],n_diseases,geometry);rng=torch.random.get_rng_state().clone()
            torch.manual_seed(20284202);model=NormalLargeCrossAttentionModel(cfg,dims,n_diseases,geometry)
            assert torch.equal(rng,torch.random.get_rng_state())
            for name,value in base.state_dict().items():assert torch.equal(value,model.state_dict()[name])
            base.train();model.train();rng=torch.random.get_rng_state().clone();out=model(x);actual=loss_fn(out,y,d,weights,cfg);actual.backward();after=torch.random.get_rng_state().clone()
            torch.random.set_rng_state(rng);original=loss_fn(base(bx),y,d,weights,cfg);original.backward()
            assert torch.equal(actual,original) and torch.equal(after,torch.random.get_rng_state()) and not torch.count_nonzero(out['large_residual_delta'])
            for name,p in base.named_parameters():assert torch.equal(p.grad,dict(model.named_parameters())[name].grad),name
            assert model.large_residual.readout.weight.grad.abs().sum()>0
            state=model.state_dict();state.update(source['state_dict']);model.load_state_dict(state,strict=True)
            with torch.no_grad():
                weight=model.large_residual.readout.weight;weight.copy_(torch.sin(torch.arange(weight.numel()).reshape(weight.shape)*.17)*.01)
                model.large_residual.readout.bias.copy_(torch.linspace(-.01,.01,256))
            model.zero_grad();plain=Model(cfg,source['dims'],n_diseases).train();state=model.state_dict(keep_vars=True)
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
            for name in ['p','large_residual_input','large_residual_hidden','large_residual_delta','large_cross_attention','large_cross_query']:
                assert torch.allclose(out[name],manual_out[name],atol=7e-6,rtol=1e-4),name
            model.eval()
            with torch.no_grad():
                attention=out['large_cross_attention'];assert torch.allclose(attention.sum(1),torch.ones(256))
                if kind=='mean':assert torch.allclose(attention,torch.full_like(attention,1/9))
                else:assert (attention-1/9).abs().max()>1e-5
                expected=model(x)['p'];permutation=torch.tensor([3,0,8,5,4,2,7,1,6]);changed={**x,'regions':x['regions'][:,permutation]}
                permutation_error=float((model(changed)['p']-expected).abs().max());assert permutation_error<4e-6
                large={k:v.detach() for k,v in model._large.items()};hidden=torch.randn(256,256)
                a=model.large_residual(large,hidden)[0];b=model.large_residual(large,hidden.roll(1,0))[0];query_change=float((a-b).abs().max())
                if kind=='mean':assert query_change==0
                else:assert query_change>1e-6
            frozen={k:v.clone() for k,v in model.named_buffers()};before=model.large_residual.value.weight.detach().clone()
            torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.1).step();assert not torch.equal(before,model.large_residual.value.weight)
            for name,value in frozen.items():assert torch.equal(value,dict(model.named_buffers())[name])
            restored=NormalLargeCrossAttentionModel(cfg,dims,n_diseases).eval();restored.load_state_dict(model.state_dict(),strict=True)
            with torch.no_grad():assert torch.equal(model(x)['p'],restored(x)['p'])
            counts.append(sum(p.numel() for p in model.parameters()));cases.append({'rank':rank,'kind':kind,'batch_size':256,'parameters':counts[-1],'maximum_complete_task_gradient_error':max(errors),'joint_region_permutation_error':permutation_error,'query_conditioning_delta_change':query_change})
        assert counts[0]==counts[1]
    cfg={**cfg,'seed':0}
    with tempfile.TemporaryDirectory(prefix='normal-large-cross-preflight-') as directory:
        folder=Path(directory);torch.manual_seed(20261124);torch.cuda.manual_seed_all(20261124)
        model=NormalLargeCrossAttentionModel(cfg,dims,n_diseases,geometry,folder/'training_input_witness.npz').cuda().train()
        local=torch.randperm(len(tf),device='cuda')[:256].cpu().numpy();gx={k:torch.from_numpy(v[local]).cuda() for k,v in data.items()}
        gy=torch.tensor(tf.iloc[local].label.to_numpy(),device='cuda');gd=torch.tensor(tf.iloc[local].disease.map({v:i for i,v in enumerate(source['diseases'])}).to_numpy(),device='cuda')
        weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[local]).cuda(),'patient_ids':torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[local]).cuda()}
        loss=cross_attention_loss(model(gx),gy,gd,weights,cfg);loss.backward();assert model.large_residual.readout.weight.grad.abs().sum()>0
        c={'cfg':cfg,'dims':dims,'diseases':source['diseases'],'stats':stats,'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}
        witness=audit_training_witness(c,folder,features,frame,fit)
    dump(ROOT/'verification_normal_large_cross_attention.json',{'passed':True,'manifest_sha256':digest,'cases':cases,'normal_geometries':diagnostics,'actual_cuda_full_task_witness':witness,
        'code_sha256':{name:sha(ROOT/name) for name in ['normal_large_cross_attention.py','audit_normal_large_cross_attention.py','verify_normal_large_cross_attention.py']},
        'checks':['Four actual256image trained-state full-task/all-parameter gradients recovered by independent NumPy normal projections and functional crossattention','Zero-init exactly restores original normalB task/base gradients and RNG; equal parameter counts across mean/attention controls','Joint regional permutation invariance; active query sensitivity only for attention; nonuniform attention after nonzero readout; optimizer update/frozen buffers/strict restore','ActualCUDA firstbatch inputs/order/labels/weights/RNG/initialparameterhashes/query/attention/residual/fulltask independently recovered'],
        'external_test_evaluated':False})
    print('Four complete-task gradient cases, identical normalB initialization/RNG, cross-query connectivity and actual CUDA first-task witness passed.',flush=True)


if __name__=='__main__':main()
