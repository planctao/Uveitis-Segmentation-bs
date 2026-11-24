"""Audit normal moments, patient weighting and actual full task updates."""
import json
import numpy as np
import pandas as pd
import torch
from experiment import ROOT,KEYS,load_data,normalize,loss_fn,sample_weights,dump
from kernel_residual_experiments import sha
from normal_subspace import NormalSuppressedModel,independent_transform
from normal_moment_geometry import fit_moment_geometry,independent_moment_geometry


def main():
    torch.set_num_threads(4)
    frame,features,digest=load_data('multiscale_vit');fit=frame.split.eq('train').to_numpy()
    normalized,_=normalize(features,fit);tf=frame.loc[fit].reset_index(drop=True)
    values={k:v[fit] for k,v in normalized.items()}
    reference=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())
    c=torch.load(reference['selected']['checkpoints'][0],map_location='cpu',weights_only=False)
    ids=np.random.default_rng(20281304).choice(len(tf),256,replace=False)
    y=torch.tensor(tf.iloc[ids].label.to_numpy());d=torch.tensor(tf.iloc[ids].disease.map({name:i for i,name in enumerate(c['diseases'])}).to_numpy())
    weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[ids]),'patient_ids':torch.from_numpy(pd.factorize(tf.patient_group)[0][ids])}
    cases=[];diagnostics={}
    for mode in ['images','within_patient']:
        geometry=fit_moment_geometry(tf,values,mode);diagnostics[mode]=independent_moment_geometry(tf,values,geometry)
        duplicate=pd.concat([tf,tf[tf.label.eq(2)]],ignore_index=True)
        duplicate_x={k:np.concatenate([v,v[tf.label.eq(2)]],0) for k,v in values.items()}
        duplicated=fit_moment_geometry(duplicate,duplicate_x,mode)
        for key in KEYS:
            rank=int(geometry[key+'_effective_rank']);a=geometry[key+'_basis'][:,:rank].astype('float64');b=duplicated[key+'_basis'][:,:rank].astype('float64')
            assert np.allclose(np.linalg.svd(a.T@b,compute_uv=False),1,atol=2e-6)
            assert np.allclose(geometry[key+'_eigenvalues'],duplicated[key+'_eigenvalues'],atol=2e-5,rtol=2e-6)
        for rank in [8,32]:
            cfg={**c['cfg'],'normal_rank':rank,'normal_strength':1.,'normal_moment_mode':mode}
            model=NormalSuppressedModel(cfg,c['dims'],len(c['diseases']),geometry)
            missing=model.load_state_dict(c['state_dict'],strict=False)
            assert len(missing.missing_keys)==8 and not missing.unexpected_keys
            x={k:torch.from_numpy(v[ids]).clone().requires_grad_(True) for k,v in values.items()}
            actual=model.transform(x)
            for key in KEYS:
                independent=independent_transform(values[key][ids],geometry[key+'_center'],geometry[key+'_basis'][:,:rank],1.)
                assert np.allclose(actual[key].detach().numpy(),independent,atol=5e-5,rtol=3e-5)
            model.train();loss=loss_fn(model(x),y,d,weights,cfg);assert torch.isfinite(loss);loss.backward()
            assert model.encoder[0].weight.grad.abs().sum()>0
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in x.values())
            torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.1).step()
            restored=NormalSuppressedModel(cfg,c['dims'],len(c['diseases']));restored.load_state_dict(model.state_dict(),strict=True)
            model.eval();restored.eval()
            with torch.no_grad():assert torch.equal(model(x)['p'],restored(x)['p'])
            cases.append(dict(mode=mode,rank=rank,real_loss_batch=256))
    # All singleton normal patients have zero within-patient covariance, so no undefined directions are removed.
    singleton=tf.drop_duplicates('patient_group').reset_index(drop=True)
    take=tf.drop_duplicates('patient_group').index.to_numpy();single_values={k:v[take] for k,v in values.items()}
    geometry=fit_moment_geometry(singleton,single_values,'within_patient')
    for key in KEYS:assert int(geometry[key+'_effective_rank'])==0 and not geometry[key+'_basis'].any()
    dump(ROOT/'verification_normal_moment_geometry.json',{'passed':True,'manifest_sha256':digest,'cases':cases,'diagnostics':diagnostics,
        'code_sha256':{name:sha(ROOT/name) for name in ['normal_moment_geometry.py','verify_normal_moment_geometry.py']},
        'checks':['Onlyfitnormalimages; eachpatientcontributes1/Nandimageswithinpatient1/n',
            'IndependentpatientloopsandsmallGram eigenproblem recoverbothmomentgeometries',
            'Duplicatingnormalimagespreservespatient-balancedspacesandspectra',
            'Degeneratewithin-patientsingletoncasehaszeropaddedbasis; noarbitrarynulldirections',
            'Allfourmodes/ranks independenttransforms andreal256-imagecompleteM5loss/update/strictrecovery']})
    print('Normal moments: weighting, independent geometry, degeneracy and complete task verified.',flush=True)


if __name__=='__main__':main()
