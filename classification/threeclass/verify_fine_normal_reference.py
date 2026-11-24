"""Verify fine region provenance, valid-patient geometry and complete tasks."""
import numpy as np
import pandas as pd
import torch
from experiment import ROOT,KEYS,Model,load_data,normalize,loss_fn,sample_weights,dump
from foreground_data import load_foreground
from normal_subspace import NormalSuppressedModel,independent_transform
from fine_normal_reference import FineNormalModel,fit_fine_geometry,independent_fine_geometry
from kernel_residual_experiments import sha


def main():
    torch.set_num_threads(4)
    reference_frame,reference_features,digest=load_data('multiscale_vit')
    source=torch.load(ROOT/'runs/normal_subspace/normal_rank32_full/validation/seed_0/model.pt',map_location='cpu',weights_only=False)
    cases=[]
    for arm in ['fine7_unmasked','fine7_masked','fine7_weighted_masked','coarse3_weighted_masked']:
        frame,features,actual_digest=load_foreground(arm);assert actual_digest==digest and frame.equals(reference_frame)
        for k in ['global','mid','dispersion']:assert np.array_equal(features[k],reference_features[k])
        for v in features.values():v[2091:]=np.nan
        fit=np.flatnonzero(frame.split.eq('train'));tf=frame.iloc[fit].reset_index(drop=True)
        mask=np.zeros(len(frame),dtype=bool);mask[fit]=True;normalized,stats=normalize(features,mask)
        values={k:v[fit] for k,v in normalized.items()};geometry=fit_fine_geometry(tf,values)
        diagnostic=independent_fine_geometry(tf,values,geometry)
        duplicate=pd.concat([tf,tf.loc[tf.label.eq(2)]],ignore_index=True)
        duplicated=fit_fine_geometry(duplicate,{k:np.concatenate([v,v[tf.label.eq(2)]]) for k,v in values.items()})
        for key in KEYS:
            assert np.allclose(np.linalg.svd(geometry[key+'_basis'].astype('float64').T@duplicated[key+'_basis'].astype('float64'),compute_uv=False),1,atol=2e-6)
        if 'region_mask' in values:
            invalid=values['region_mask']<.5;changed={k:v.copy() for k,v in values.items()};changed['regions'][invalid]=1e5
            ignored=fit_fine_geometry(tf,changed)
            for key in KEYS:assert np.array_equal(geometry[key+'_basis'],ignored[key+'_basis'])
        cfg={**source['cfg'],'fine_normal_arm':arm,'use_region_mask':'region_mask' in features}
        dims={**source['dims'],**({'region_mask':features['region_mask'].shape[1]} if 'region_mask' in features else {})}
        torch.manual_seed(20282201);baseline=Model(cfg,dims,len(source['diseases']));rng=torch.random.get_rng_state().clone()
        torch.manual_seed(20282201);model=FineNormalModel(cfg,dims,len(source['diseases']),geometry)
        assert torch.equal(rng,torch.random.get_rng_state())
        for k,v in baseline.state_dict().items():assert torch.equal(v,model.state_dict()[k])
        model.load_state_dict({k:v for k,v in source['state_dict'].items() if not k.startswith('normal_')},strict=False)
        ids=np.random.default_rng(20282202).choice(len(tf),256,replace=False)
        x={k:torch.from_numpy(v[ids]).clone() for k,v in values.items()}
        expected={k:torch.from_numpy(independent_transform(values[k][ids],geometry[k+'_center'],geometry[k+'_basis'],1.)) for k in KEYS}
        if 'region_mask' in x:expected['region_mask']=x['region_mask']
        plain=Model(cfg,dims,len(source['diseases']));plain.load_state_dict({k:v for k,v in model.state_dict().items() if not k.startswith('normal_')},strict=True)
        model.eval();plain.eval()
        with torch.no_grad():
            assert torch.allclose(model(x)['p'],plain(expected)['p'],atol=4e-6,rtol=4e-5)
            assert torch.equal(model(x)['p'].argmax(1),plain(expected)['p'].argmax(1))
        model.cfg['normal_strength']=0.
        with torch.no_grad():assert torch.equal(model(x)['p'],plain(x)['p'])
        model.cfg['normal_strength']=1.
        if 'region_mask' in x:
            a=model(x)['p'];changed={k:v.clone() for k,v in x.items()};changed['regions'][changed['region_mask']<.5]=1e4
            assert torch.equal(a,model(changed)['p'])
        y=torch.tensor(tf.iloc[ids].label.to_numpy());d=torch.tensor(tf.iloc[ids].disease.map({v:i for i,v in enumerate(source['diseases'])}).to_numpy())
        weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[ids]),'patient_ids':torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[ids])}
        tx={k:v.clone().requires_grad_(k!='region_mask') for k,v in x.items()};model.train()
        loss=loss_fn(model(tx),y,d,weights,cfg);assert torch.isfinite(loss);loss.backward()
        assert model.encoder[0].weight.grad.abs().sum()>0
        assert all(tx[k].grad is not None and torch.isfinite(tx[k].grad).all() for k in KEYS)
        before=model.encoder[0].weight.detach().clone();torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.1).step()
        assert not torch.equal(before,model.encoder[0].weight)
        restored=FineNormalModel(cfg,dims,len(source['diseases']));restored.load_state_dict(model.state_dict(),strict=True)
        model.eval();restored.eval()
        with torch.no_grad():assert torch.equal(restored(x)['p'],model(x)['p'])
        cases.append({'arm':arm,'region_tokens':features['regions'].shape[1],'full_task_batch':256,'geometry':diagnostic})
        del features,normalized,values
    dump(ROOT/'verification_fine_normal_reference.json',{'passed':True,'manifest_sha256':digest,'cases':cases,
        'code_sha256':{n:sha(ROOT/n) for n in ['fine_normal_reference.py','verify_fine_normal_reference.py']},
        'checks':['Existing verified3/7grid sourcebank identity; all original global/mid/dispersion arrays preserved exactly',
            'Fit-normal patient PCA uses image means over valid regions only; independent patient loops/Gram spectra recover every space',
            'Normal image duplication and invalid-token changes leave geometry unchanged; masked predictions ignore invalid regions',
            'Original complete parameter/RNG identity,zero-strengthplainMLPidentity and independent NumPy projections verified',
            'Four real256-image complete task gradient/update and strict recovery checks']})
    print('Fine normal-reference geometry,mask invariants and4fulltasks verified.',flush=True)


if __name__=='__main__':main()
