"""Full-task functional gradients and D4 invariance before formal training."""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment import ROOT,Model,load_data,normalize,loss_fn,sample_weights,dump
from normal_subspace import NormalSuppressedModel,fit_normal_geometry
from normal_spatial_contrast import NormalSpatialContrastModel,spatial_contrast_loss,d4_permutations
from audit_normal_spatial_contrast import independent_forward,manual_branch,audit_training_witness
from audit_normal_tangent import complete_task
from verify_normal_subspace import independent_geometry
from kernel_residual_experiments import sha
from dinov3_large_data import load_derived


def main():
    torch.set_num_threads(4)
    frame=pd.read_csv(ROOT/'metadata/manifest.csv');digest=sha(ROOT/'metadata/manifest.csv')
    features=load_derived(ROOT,frame,digest,'base_dual_large_dual')
    for v in features.values():v[2091:]=np.nan
    fit=np.flatnonzero(frame.split.eq('train'));mask=np.zeros(len(frame),bool);mask[fit]=True
    normalized,stats=normalize(features,mask);tf=frame.iloc[fit].reset_index(drop=True);data={k:v[fit] for k,v in normalized.items()}
    geometry=fit_normal_geometry(tf,data);diagnostic=independent_geometry(tf,data,geometry)
    source=torch.load(ROOT/'runs/normal_large_reference/base_dual_large_dual/validation/seed_0/model.pt',map_location='cpu',weights_only=False)
    ids=np.random.default_rng(20283401).choice(len(tf),256,replace=False);raw={k:v[ids] for k,v in data.items()};x={k:torch.from_numpy(v) for k,v in raw.items()}
    y=torch.tensor(tf.iloc[ids].label.to_numpy());d=torch.tensor(tf.iloc[ids].disease.map({v:i for i,v in enumerate(source['diseases'])}).to_numpy())
    weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[ids]),'patient_ids':torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[ids])};cases=[]
    assert len({tuple(v) for v in d4_permutations()})==8
    for rank in [16,32]:
        parameter_counts=[]
        for kind in ['mean','d4']:
            cfg={**source['cfg'],'seed':0,'spatial_contrast_kind':kind,'spatial_contrast_rank':rank}
            torch.manual_seed(20283402);base=NormalSuppressedModel(cfg.copy(),source['dims'].copy(),len(source['diseases']),geometry);rng=torch.random.get_rng_state().clone()
            torch.manual_seed(20283402);model=NormalSpatialContrastModel(cfg.copy(),source['dims'].copy(),len(source['diseases']),geometry)
            assert torch.equal(rng,torch.random.get_rng_state())
            for k,v in base.state_dict().items():assert torch.equal(v,model.state_dict()[k])
            parameter_counts.append(sum(v.numel() for v in model.parameters()))
            model.train();base.train();rng=torch.random.get_rng_state().clone();out=model(x);actual=loss_fn(out,y,d,weights,cfg);actual.backward();after=torch.random.get_rng_state().clone()
            torch.random.set_rng_state(rng);original=loss_fn(base(x),y,d,weights,cfg);original.backward()
            assert torch.equal(actual,original) and torch.equal(after,torch.random.get_rng_state()) and torch.count_nonzero(out['spatial_delta'])==0
            for k,p in base.named_parameters():assert torch.equal(p.grad,dict(model.named_parameters())[k].grad),k
            assert model.spatial_contrast.readout.weight.grad.abs().sum()>0
            # Activate the readout deterministically and use trained original parameters.
            state=model.state_dict();state.update(source['state_dict']);model.load_state_dict(state,strict=True)
            with torch.no_grad():
                w=model.spatial_contrast.readout.weight;w.copy_(torch.sin(torch.arange(w.numel()).reshape(w.shape)*.17)*.01)
                model.spatial_contrast.readout.bias.copy_(torch.linspace(-.01,.01,256))
            model.zero_grad();model.train();plain=Model(cfg,source['dims'],len(source['diseases'])).train()
            state=model.state_dict(keep_vars=True);plain.load_state_dict({k:v for k,v in state.items() if not k.startswith(('normal_','spatial_contrast.'))},strict=True)
            independent_state={k:(v.detach().clone().requires_grad_(True) if k.startswith('spatial_contrast.') and v.is_floating_point() else v.detach().clone()) for k,v in state.items()}
            rng=torch.random.get_rng_state().clone();out=model(x);actual=loss_fn(out,y,d,weights,cfg);actual.backward();after=torch.random.get_rng_state().clone()
            torch.random.set_rng_state(rng);replayed=independent_forward(plain,raw,independent_state,cfg);manual=complete_task(replayed,y,d,weights,cfg);manual.backward()
            assert torch.equal(after,torch.random.get_rng_state()) and torch.allclose(actual,manual,atol=6e-6,rtol=6e-5)
            errors={}
            for k,p in model.named_parameters():
                independent=independent_state[k].grad if k.startswith('spatial_contrast.') else dict(plain.named_parameters())[k].grad
                assert independent is not None and torch.allclose(p.grad,independent,atol=1e-5,rtol=2e-3),k
                errors[k]=float((p.grad-independent).abs().max())
            for key in ['p','spatial_delta','spatial_values','spatial_summary']:assert torch.allclose(out[key],replayed[key],atol=7e-6,rtol=1e-4),key
            # Group invariance is tested after nonzero readout, including original attention.
            model.eval()
            with torch.no_grad():
                expected=model(x)['p'];maximum=0.
                for perm in d4_permutations():
                    other=model({**x,'regions':x['regions'][:,perm]})['p'];maximum=max(maximum,float((expected-other).abs().max()))
                    assert torch.allclose(expected,other,atol=3e-6,rtol=3e-5)
                tokens=out['spatial_tokens'].detach();delta=model.spatial_contrast(tokens)[0]
                arbitrary=torch.tensor([0,4,2,3,1,5,6,7,8]);changed=model.spatial_contrast(tokens[:,arbitrary])[0]
                if kind=='mean':assert torch.allclose(delta,changed,atol=3e-6,rtol=3e-5)
                else:assert float((delta-changed).abs().max())>1e-6
                assert torch.allclose(model.spatial_contrast(tokens)[0],model.spatial_contrast(tokens+tokens.new_tensor(.25))[0],atol=4e-6,rtol=4e-5)
            frozen={k:v.clone() for k,v in model.named_buffers()};before=model.spatial_contrast.down.weight.detach().clone()
            torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.1).step();assert not torch.equal(before,model.spatial_contrast.down.weight)
            for k,v in frozen.items():assert torch.equal(v,dict(model.named_buffers())[k])
            restored=NormalSpatialContrastModel(cfg,source['dims'],len(source['diseases'])).eval();restored.load_state_dict(model.state_dict(),strict=True)
            with torch.no_grad():assert torch.equal(model(x)['p'],restored(x)['p'])
            cases.append({'kind':kind,'rank':rank,'batch_size':256,'parameters':parameter_counts[-1],'maximum_full_task_gradient_error':max(errors.values()),'maximum_d4_invariance_error':maximum})
        assert parameter_counts[0]==parameter_counts[1]
    cfg={**source['cfg'],'seed':0,'spatial_contrast_kind':'d4','spatial_contrast_rank':32}
    with tempfile.TemporaryDirectory(prefix='spatial-contrast-preflight-') as directory:
        folder=Path(directory);torch.manual_seed(20261124);torch.cuda.manual_seed_all(20261124)
        model=NormalSpatialContrastModel(cfg.copy(),source['dims'].copy(),len(source['diseases']),geometry,folder/'training_input_witness.npz').cuda().train()
        local=torch.randperm(len(tf),device='cuda')[:256].cpu().numpy();x={k:torch.from_numpy(v[local]).cuda() for k,v in data.items()}
        y=torch.tensor(tf.iloc[local].label.to_numpy(),device='cuda');d=torch.tensor(tf.iloc[local].disease.map({v:i for i,v in enumerate(source['diseases'])}).to_numpy(),device='cuda')
        weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[local]).cuda(),'patient_ids':torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[local]).cuda()}
        loss=spatial_contrast_loss(model(x),y,d,weights,cfg);loss.backward();assert model.spatial_contrast.readout.weight.grad.abs().sum()>0
        c={'cfg':cfg,'dims':source['dims'],'diseases':source['diseases'],'stats':stats,'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}
        cuda_proof=audit_training_witness(c,folder,features,frame,fit)
    dump(ROOT/'verification_large_spatial_contrast.json',{'passed':True,'manifest_sha256':digest,'cases':cases,'normal_geometry':diagnostic,'cuda_full_task_witness':cuda_proof,
        'code_sha256':{p:sha(ROOT/p) for p in ['normal_spatial_contrast.py','audit_normal_spatial_contrast.py','verify_large_spatial_contrast.py']},
        'checks':['All4full256imagetrainedstates/completeM5task/allparametergradients match independentfunctionalbranchandNumPynormalprojection',
            'Exactoriginalinit/outputs/basegradients/RNGatzeroreadout;newreadoutgradientnonzero;newconstructionpreservesoriginaldrawstream',
            'D4invariancewithnonzerobranch,meancontrolarbitrarypermutationinvariance,D4noninvariancetoarbitrarypermutation,commonshiftinvarianceofcontrasttokens',
            'Equalparamcountcontrols,optimizerupdates,strictrestores,frozengeometryandpermutations;actualCUDAfirstfulltaskindependentlyrestored']})
    print('Large spatial contrast:4fulltaskgradients,neutralinit,D4invariance,capacitycontrolandCUDAwitness passed.',flush=True)


if __name__=='__main__':main()
