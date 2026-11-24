"""Verify selective projection, unchanged other streams and actual complete M5 updates."""
import json
import numpy as np
import pandas as pd
import torch
from experiment import ROOT,KEYS,load_data,normalize,loss_fn,sample_weights,dump
from kernel_residual_experiments import sha
from normal_subspace import fit_normal_geometry,independent_transform
from scoped_normal_subspace import ScopedNormalModel,SCOPES


def main():
    torch.set_num_threads(4)
    frame,features,digest=load_data('multiscale_vit');fit=frame.split.eq('train').to_numpy()
    normalized,_=normalize(features,fit);tf=frame.loc[fit].reset_index(drop=True)
    values={k:v[fit] for k,v in normalized.items()};geometry=fit_normal_geometry(tf,values)
    reference=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())
    c=torch.load(reference['selected']['checkpoints'][0],map_location='cpu',weights_only=False)
    ids=np.random.default_rng(20281303).choice(len(tf),256,replace=False)
    y=torch.tensor(tf.iloc[ids].label.to_numpy());d=torch.tensor(tf.iloc[ids].disease.map({v:i for i,v in enumerate(c['diseases'])}).to_numpy())
    weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[ids]),'patient_ids':torch.from_numpy(pd.factorize(tf.patient_group)[0][ids])}
    cases=[]
    for scope,keys in SCOPES.items():
        cfg={**c['cfg'],'normal_rank':32,'normal_strength':1.,'normal_scope':scope}
        model=ScopedNormalModel(cfg,c['dims'],len(c['diseases']),geometry)
        missing=model.load_state_dict(c['state_dict'],strict=False)
        assert not missing.unexpected_keys and len(missing.missing_keys)==8 and all(k.startswith('normal_') for k in missing.missing_keys)
        x={k:torch.from_numpy(v[ids]).clone().requires_grad_(True) for k,v in values.items()}
        transformed=model.transform(x)
        for key in KEYS:
            if key not in keys:assert transformed[key] is x[key]
            else:
                independent=independent_transform(values[key][ids],geometry[key+'_center'],geometry[key+'_basis'],1.)
                assert np.allclose(transformed[key].detach().numpy(),independent,atol=4e-5,rtol=3e-5)
        frozen={k:v.clone() for k,v in model.named_buffers() if k.startswith('normal_')}
        model.train();out=model(x);loss=loss_fn(out,y,d,weights,cfg);assert torch.isfinite(loss);loss.backward()
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in x.values())
        assert model.encoder[0].weight.grad.abs().sum()>0
        torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.1).step()
        assert all(torch.equal(v,dict(model.named_buffers())[k]) for k,v in frozen.items())
        restored=ScopedNormalModel(cfg,c['dims'],len(c['diseases']));restored.load_state_dict(model.state_dict(),strict=True)
        model.eval();restored.eval()
        with torch.no_grad():assert torch.equal(model(x)['p'],restored(x)['p'])
        cases.append({'scope':scope,'projected_keys':keys,'real_loss_batch':256,'parameters':sum(v.numel() for v in model.parameters())})
    dump(ROOT/'verification_scoped_normal_subspace.json',{'passed':True,'manifest_sha256':digest,'cases':cases,
        'code_sha256':{name:sha(ROOT/name) for name in ['scoped_normal_subspace.py','verify_scoped_normal_subspace.py']},
        'checks':['Fourpredeclaredscopes affectexactlytheirrequestedstreams; otherinputtensors identical',
            'IndependentNumPyprojectionmatchesallactualselectiveoutputs',
            'Eachscopeusesreal256-imagefullM5task/backward andsuccessfulMLPupdate',
            'Allnormalgeometrybuffersremainfrozen; strictcheckpointrestoration matchespredictions']})
    print('Scoped normal projection: exact stream scope, independent transforms and full task verified.',flush=True)


if __name__=='__main__':main()
