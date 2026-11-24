"""Normal-only atlas geometry, conditional projection and complete-task checks."""
import numpy as np
import pandas as pd
import torch
from experiment import ROOT,KEYS,Model,load_data,normalize,loss_fn,sample_weights,dump
from normal_atlas import NormalAtlasModel,fit_atlas_geometry,independent_geometry,independent_transform,numpy_gate
from kernel_residual_experiments import sha


def main():
    torch.set_num_threads(4)
    frame,features,digest=load_data('multiscale_vit')
    for value in features.values():value[2091:]=np.nan
    fit=np.flatnonzero(frame.split.eq('train'));mask=np.zeros(len(frame),bool);mask[fit]=True
    normalized,_=normalize(features,mask);tf=frame.iloc[fit].reset_index(drop=True);data={k:v[fit] for k,v in normalized.items()}
    source=torch.load(ROOT/'runs/normal_subspace/normal_rank32_full/validation/seed_0/model.pt',map_location='cpu',weights_only=False)
    cases=[]
    for count in [2,4]:
        for temperature in [.5,2.]:
            geometry=fit_atlas_geometry(tf,data,count,temperature)
            diagnostic=independent_geometry(tf,data,geometry,count,temperature)
            duplicate=pd.concat([tf,tf.loc[tf.label.eq(2)]],ignore_index=True)
            doubled=fit_atlas_geometry(duplicate,{k:np.concatenate([v,v[tf.label.eq(2)]]) for k,v in data.items()},count,temperature)
            changed={k:v.copy() for k,v in data.items()}
            for k in KEYS:changed[k][tf.label.ne(2)]=7.25
            unaffected=fit_atlas_geometry(tf,changed,count,temperature)
            for key in geometry:
                if key=='normal_patients':
                    assert np.array_equal(geometry[key],doubled[key]) and np.array_equal(geometry[key],unaffected[key]);continue
                assert np.array_equal(geometry[key],unaffected[key]),key
                assert np.allclose(geometry[key],doubled[key],atol=3e-6,rtol=3e-6),key
            cfg={**source['cfg'],'normal_atlas_count':count,'normal_atlas_temperature':temperature,'normal_atlas_strength':1.}
            cfg.pop('normal_rank');cfg.pop('normal_strength')
            torch.manual_seed(20282401);plain=Model(cfg.copy(),source['dims'],len(source['diseases']));rng=torch.random.get_rng_state().clone()
            torch.manual_seed(20282401);model=NormalAtlasModel(cfg.copy(),source['dims'],len(source['diseases']),geometry)
            assert torch.equal(rng,torch.random.get_rng_state())
            assert dict(model.named_parameters()).keys()==dict(plain.named_parameters()).keys()
            for key,v in plain.state_dict().items():assert torch.equal(v,model.state_dict()[key])
            ids=np.random.default_rng(20282402).choice(len(tf),256,replace=False)
            x={k:torch.from_numpy(v[ids]).clone() for k,v in data.items()}
            expected=independent_transform({k:v[ids] for k,v in data.items()},geometry,temperature)
            independent={k:torch.from_numpy(v) for k,v in expected.items()}
            model.eval();plain.eval()
            with torch.no_grad():
                transformed=model.transform(x)
                for key in KEYS:assert np.allclose(transformed[key],expected[key],atol=5e-6,rtol=5e-5),key
                weights,_=numpy_gate(data['global'][ids],geometry,temperature)
                assert np.allclose(weights,model._weights,atol=2e-6,rtol=2e-6)
                assert torch.allclose(model(x)['p'],plain(independent)['p'],atol=3e-6,rtol=3e-5)
            y=torch.tensor(tf.iloc[ids].label.to_numpy());d=torch.tensor(tf.iloc[ids].disease.map({v:i for i,v in enumerate(source['diseases'])}).to_numpy())
            weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[ids]),'patient_ids':torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[ids])}
            model.train();plain.train();rng=torch.random.get_rng_state().clone()
            loss=loss_fn(model(x),y,d,weights,cfg);loss.backward();after=torch.random.get_rng_state().clone()
            torch.random.set_rng_state(rng);manual=loss_fn(plain(independent),y,d,weights,cfg);manual.backward()
            assert torch.equal(after,torch.random.get_rng_state()) and torch.allclose(loss,manual,atol=3e-6,rtol=3e-5)
            for key,p in model.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all()
                assert torch.allclose(p.grad,dict(plain.named_parameters())[key].grad,atol=3e-6,rtol=5e-4),key
            before=model.encoder[0].weight.detach().clone();torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.1).step()
            assert not torch.equal(before,model.encoder[0].weight)
            restored=NormalAtlasModel(cfg.copy(),source['dims'],len(source['diseases']));restored.load_state_dict(model.state_dict(),strict=True)
            model.eval();restored.eval()
            with torch.no_grad():assert torch.equal(model(x)['p'],restored(x)['p'])
            # Disable only atlas transformation: ordinary M5 full loss,gradients,and RNG are exact.
            plain.load_state_dict({k:v for k,v in model.state_dict().items() if not k.startswith('atlas_')},strict=True)
            model.cfg['normal_atlas_strength']=0.;model.train();plain.train();model.zero_grad();plain.zero_grad()
            rng=torch.random.get_rng_state().clone();loss=loss_fn(model(x),y,d,weights,cfg);loss.backward();after=torch.random.get_rng_state().clone()
            torch.random.set_rng_state(rng);manual=loss_fn(plain(x),y,d,weights,cfg);manual.backward()
            assert torch.equal(after,torch.random.get_rng_state()) and torch.equal(loss,manual)
            for key,p in model.named_parameters():assert torch.equal(p.grad,dict(plain.named_parameters())[key].grad),key
            # Input derivatives through conditional routing remain finite and nonzero.
            model.cfg['normal_atlas_strength']=1.;model.eval();model.zero_grad()
            differentiable={k:v.clone().requires_grad_() for k,v in x.items()}
            objective=loss_fn(model(differentiable),y,d,weights,cfg);objective.backward()
            for key in KEYS:assert torch.isfinite(differentiable[key].grad).all() and differentiable[key].grad.abs().sum()>0
            cases.append({'clusters':count,'temperature':temperature,'full_task_batch':256,'geometry':diagnostic})
    dump(ROOT/'verification_normal_atlas.json',{'passed':True,'manifest_sha256':digest,'cases':cases,
        'code_sha256':{name:sha(ROOT/name) for name in ['normal_atlas.py','verify_normal_atlas.py']},
        'checks':['All geometry uses normal patient centers only; duplicated normal images and changed nonnormal images preserve geometry',
            'Global gate and each weighted local tangent recovered independently by Gram eigenproblems; clustering fixed-point, responsibility and shrinkage weights audited',
            'Independent NumPy conditional routing/projections plus plain M5 reproduce input transforms, probabilities and full task parameter gradients',
            'Original parameter schema/initialization/RNG unchanged; disabled atlas exact full-task loss,gradient and RNG identity',
            'Four real256image full task gradient/update/strict checkpoint recovery checks and differentiable input routing passed']})
    print('Normal atlas: patient geometry,conditional routing and4fulltasks verified.',flush=True)


if __name__=='__main__':main()
