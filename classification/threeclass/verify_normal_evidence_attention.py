"""Verify normal evidence statistics, attention identities and full M5 tasks."""
import numpy as np
import pandas as pd
import torch
from experiment import ROOT,KEYS,Model,load_data,normalize,loss_fn,sample_weights,dump
from normal_subspace import NormalSuppressedModel,independent_transform
from normal_evidence_attention import NormalEvidenceModel,fit_evidence_geometry,independent_geometry,numpy_bias
from kernel_residual_experiments import sha


def main():
    torch.set_num_threads(4)
    frame,features,digest=load_data('multiscale_vit')
    for v in features.values():v[2091:]=np.nan
    fit=np.flatnonzero(frame.split.eq('train'));mask=np.zeros(len(frame),bool);mask[fit]=True
    normalized,_=normalize(features,mask);tf=frame.iloc[fit].reset_index(drop=True);data={k:v[fit] for k,v in normalized.items()}
    source=torch.load(ROOT/'runs/normal_subspace/normal_rank32_full/validation/seed_0/model.pt',map_location='cpu',weights_only=False)
    cases=[]
    for scope in ['pooled','position']:
        geometry=fit_evidence_geometry(tf,data,scope);diagnostic=independent_geometry(tf,data,geometry,scope)
        duplicate=pd.concat([tf,tf.loc[tf.label.eq(2)]],ignore_index=True)
        doubled=fit_evidence_geometry(duplicate,{k:np.concatenate([v,v[tf.label.eq(2)]]) for k,v in data.items()},scope)
        changed={k:v.copy() for k,v in data.items()}
        for k in KEYS:changed[k][tf.label.ne(2)]=7.25
        unaffected=fit_evidence_geometry(tf,changed,scope)
        for k in KEYS:
            assert np.allclose(np.linalg.svd(geometry[k+'_basis'].astype('float64').T@doubled[k+'_basis'].astype('float64'),compute_uv=False),1,atol=2e-6)
            assert np.array_equal(geometry[k+'_basis'],unaffected[k+'_basis'])
        for key in ['evidence_center','evidence_scale']:
            assert np.allclose(geometry[key],doubled[key],atol=1e-6,rtol=1e-6)
            assert np.array_equal(geometry[key],unaffected[key])
        for strength in [.5,2.]:
            cfg={**source['cfg'],'normal_evidence_scope':scope,'normal_evidence_strength':strength}
            torch.manual_seed(20282301);baseline=NormalSuppressedModel(cfg,source['dims'],len(source['diseases']),geometry);rng=torch.random.get_rng_state().clone()
            torch.manual_seed(20282301);model=NormalEvidenceModel(cfg.copy(),source['dims'],len(source['diseases']),geometry)
            assert torch.equal(rng,torch.random.get_rng_state())
            assert dict(model.named_parameters()).keys()==dict(baseline.named_parameters()).keys()
            for k,v in baseline.state_dict().items():assert torch.equal(v,model.state_dict()[k])
            ids=np.random.default_rng(20282302).choice(len(tf),256,replace=False)
            x={k:torch.from_numpy(v[ids]).clone() for k,v in data.items()}
            expected={k:torch.from_numpy(independent_transform(data[k][ids],geometry[k+'_center'],geometry[k+'_basis'],1.)) for k in KEYS}
            energy,bias=numpy_bias(expected['regions'].numpy(),geometry['evidence_center'],geometry['evidence_scale'],strength)
            plain=Model(cfg.copy(),source['dims'],len(source['diseases']))
            plain.load_state_dict({k:v for k,v in model.state_dict().items() if not k.startswith('normal_') and not k.startswith('evidence_')},strict=True)
            hook=plain.region_gate.register_forward_hook(lambda module,args,output:output+torch.from_numpy(bias)[:,:,None])
            model.eval();plain.eval();baseline.eval()
            with torch.no_grad():
                actual=model(x);manual=plain(expected)
                assert torch.allclose(actual['p'],manual['p'],atol=3e-6,rtol=3e-5)
                assert np.allclose(actual['normal_region_energy'],energy,atol=2e-6,rtol=2e-6)
                assert np.allclose(actual['normal_attention_bias'],bias,atol=2e-6,rtol=2e-6)
                assert torch.max(torch.abs(actual['normal_attention_bias'].mean(1)))<1e-6
                assert torch.max(torch.abs(actual['normal_attention_bias']))<=2*strength+1e-6
                assert np.std(bias,axis=1).mean()>1e-3
                assert torch.allclose(actual['normal_region_attention'].sum(1),torch.ones(256,2),atol=1e-6)
                # Token order invariance; position references must travel with their tokens.
                permutation=torch.tensor([8,2,5,0,6,1,7,3,4]);xp={k:v.clone() for k,v in x.items()};xp['regions']=xp['regions'][:,permutation]
                saved_center=model.evidence_center.clone();saved_scale=model.evidence_scale.clone()
                if scope=='position':model.evidence_center.copy_(saved_center[permutation]);model.evidence_scale.copy_(saved_scale[permutation])
                assert torch.allclose(model(xp)['p'],actual['p'],atol=2e-6,rtol=2e-5)
                model.evidence_center.copy_(saved_center);model.evidence_scale.copy_(saved_scale)
            y=torch.tensor(tf.iloc[ids].label.to_numpy());d=torch.tensor(tf.iloc[ids].disease.map({v:i for i,v in enumerate(source['diseases'])}).to_numpy())
            weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[ids]),'patient_ids':torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[ids])}
            model.train();plain.train();rng=torch.random.get_rng_state().clone()
            out=model(x);loss=loss_fn(out,y,d,weights,cfg);loss.backward();after=torch.random.get_rng_state().clone()
            torch.random.set_rng_state(rng);independent_loss=loss_fn(plain(expected),y,d,weights,cfg);independent_loss.backward()
            assert torch.equal(after,torch.random.get_rng_state()) and torch.allclose(loss,independent_loss,atol=3e-6,rtol=3e-5)
            for k,p in model.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all()
                assert torch.allclose(p.grad,dict(plain.named_parameters())[k].grad,atol=3e-6,rtol=5e-4),k
            before=model.encoder[0].weight.detach().clone();torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.1).step()
            assert not torch.equal(before,model.encoder[0].weight)
            restored=NormalEvidenceModel(cfg.copy(),source['dims'],len(source['diseases']));restored.load_state_dict(model.state_dict(),strict=True)
            model.eval();restored.eval()
            with torch.no_grad():assert torch.equal(model(x)['p'],restored(x)['p'])
            # Zero prior restores normal-reference model including complete losses/gradients/RNG.
            baseline.load_state_dict({k:v for k,v in model.state_dict().items() if not k.startswith('evidence_')},strict=True)
            model.cfg['normal_evidence_strength']=0.;model.train();baseline.train();model.zero_grad();baseline.zero_grad()
            rng=torch.random.get_rng_state().clone();zero=loss_fn(model(x),y,d,weights,cfg);zero.backward();after=torch.random.get_rng_state().clone()
            torch.random.set_rng_state(rng);old=loss_fn(baseline(x),y,d,weights,cfg);old.backward()
            assert torch.equal(after,torch.random.get_rng_state()) and torch.equal(zero,old)
            for k,p in model.named_parameters():assert torch.equal(p.grad,dict(baseline.named_parameters())[k].grad),k
            hook.remove()
            cases.append({'scope':scope,'strength':strength,'full_task_batch':256,'mean_bias_sd':float(np.std(bias,axis=1).mean()),'geometry':diagnostic})
    dump(ROOT/'verification_normal_evidence_attention.json',{'passed':True,'manifest_sha256':digest,'cases':cases,
        'code_sha256':{name:sha(ROOT/name) for name in ['normal_evidence_attention.py','verify_normal_evidence_attention.py']},
        'checks':['Fit-only normal patient PCA independently recovered by Gram eigensystems',
            'Patient-balanced regional moments independently recovered by centered second pass; normal duplication/non-normal changes leave geometry unchanged',
            'Same original trainable parameter schema, initialization and RNG; zero prior exactly restores complete task loss/gradients/RNG',
            'Independent NumPy projection, energies, biases and ordinary MLP reproduce predictions; consistent token permutation preserves probabilities',
            'Four real256image full tasks with independent losses/parameter gradients, updates and strict checkpoint recovery']})
    print('Normal energy attention: geometry,identities,and4fulltasks verified.',flush=True)


if __name__=='__main__':main()
