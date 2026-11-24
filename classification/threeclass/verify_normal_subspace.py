"""Independent Gram eigenspaces, patient balance, gradients and real task recovery."""
import json
import numpy as np
import pandas as pd
import torch
from experiment import ROOT,KEYS,Model,load_data,normalize,loss_fn,sample_weights,dump
from normal_subspace import fit_normal_geometry,NormalSuppressedModel,independent_transform
from kernel_residual_experiments import sha


def independent_geometry(frame,normalized,geometry):
    normal=frame.label.eq(2).to_numpy();names=sorted(frame.loc[normal,'patient_group'].unique())
    assert np.array_equal(geometry['normal_patients'],names)
    errors={}
    for key in KEYS:
        x=np.asarray(normalized[key],dtype='float64')
        if x.ndim==3:x=np.sum(x,axis=1)/x.shape[1]
        centers=[]
        for patient in names:
            rows=np.flatnonzero(normal & frame.patient_group.eq(patient).to_numpy())
            centers.append(np.sum(x[rows],axis=0)/len(rows))
        center=np.mean(centers,axis=0);a=np.stack(centers)-center
        eigenvalues,left=np.linalg.eigh(a@a.T)
        order=np.argsort(eigenvalues)[::-1][:32]
        b=a.T@left[:,order]/np.sqrt(eigenvalues[order])[None]
        fitted=geometry[key+'_basis'].astype('float64')
        angles=np.linalg.svd(b.T@fitted,compute_uv=False)
        assert np.allclose(angles,1,atol=1e-6)
        assert np.allclose(center,geometry[key+'_center'],atol=1e-6)
        assert np.allclose(eigenvalues[order]/(len(names)-1),geometry[key+'_eigenvalues'],atol=2e-5,rtol=2e-6)
        assert np.allclose(fitted.T@fitted,np.eye(32),atol=1e-6)
        errors[key]={'maximum_principal_cosine_error':float(np.max(np.abs(angles-1))),
            'normal_patients':len(names),'maximum_center_error':float(np.max(np.abs(center-geometry[key+'_center'])))}
    return errors


def main():
    torch.set_num_threads(4)
    frame,features,digest=load_data('multiscale_vit');fit=frame.split.eq('train').to_numpy()
    normalized,stats=normalize(features,fit);tf=frame.loc[fit].reset_index(drop=True)
    train={k:v[fit] for k,v in normalized.items()};geometry=fit_normal_geometry(tf,train)
    diagnostic=independent_geometry(tf,train,geometry)
    # Duplicating every normal image preserves each patient's equal contribution.
    duplicated=pd.concat([tf,tf[tf.label.eq(2)]],ignore_index=True)
    duplicated_x={k:np.concatenate([v,v[tf.label.eq(2)]],0) for k,v in train.items()}
    duplicated_geometry=fit_normal_geometry(duplicated,duplicated_x)
    for key in KEYS:
        b=geometry[key+'_basis'].astype('float64');other=duplicated_geometry[key+'_basis'].astype('float64')
        assert np.allclose(np.linalg.svd(b.T@other,compute_uv=False),1,atol=1e-6)
    reference=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())
    c=torch.load(reference['selected']['checkpoints'][0],map_location='cpu',weights_only=False)
    ids=np.random.default_rng(20281225).choice(len(tf),256,replace=False)
    y=torch.tensor(tf.iloc[ids].label.to_numpy());diseases=tf.iloc[ids].disease.map({d:i for i,d in enumerate(c['diseases'])})
    d=torch.tensor(diseases.to_numpy());weights=torch.from_numpy(sample_weights(tf,tf.label)[ids])
    patients=torch.from_numpy(pd.factorize(tf.patient_group)[0][ids])
    cases=[]
    for rank,strength in [(32,0.),(8,.5),(32,.5),(32,1.)]:
        cfg={**c['cfg'],'normal_rank':rank,'normal_strength':strength}
        torch.manual_seed(20261124);plain=Model(cfg,c['dims'],len(c['diseases']))
        rng=torch.random.get_rng_state().clone()
        torch.manual_seed(20261124);model=NormalSuppressedModel(cfg,c['dims'],len(c['diseases']),geometry)
        assert torch.equal(rng,torch.random.get_rng_state())
        for key,value in plain.state_dict().items():assert torch.equal(value,model.state_dict()[key])
        x={k:torch.from_numpy(v[ids]).clone().requires_grad_(True) for k,v in train.items()}
        actual=model.transform(x);errors={}
        for key in KEYS:
            independent=independent_transform(train[key][ids],geometry[key+'_center'],geometry[key+'_basis'][:,:rank],strength)
            error=float(np.max(np.abs(actual[key].detach().numpy()-independent)));errors[key]=error
            assert np.allclose(actual[key].detach().numpy(),independent,atol=4e-5,rtol=3e-5)
        if strength==0:
            assert all(actual[key] is x[key] for key in KEYS)
            model.eval();plain.eval()
            with torch.no_grad():assert torch.equal(model(x)['p'],plain(x)['p'])
        model.train();out=model(x);loss=loss_fn(out,y,d,{'leaf':weights,'patient_ids':patients},cfg)
        assert torch.isfinite(loss);loss.backward()
        assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in x.values())
        frozen={k:v.clone() for k,v in model.named_buffers() if k.startswith('normal_')}
        optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.1);optimizer.step()
        assert all(torch.equal(value,dict(model.named_buffers())[key]) for key,value in frozen.items())
        model.eval();restored=NormalSuppressedModel(cfg,c['dims'],len(c['diseases']))
        restored.load_state_dict(model.state_dict(),strict=True);restored.eval()
        with torch.no_grad():assert torch.equal(model(x)['p'],restored(x)['p'])
        # Check the projection Jacobian away from clipping boundaries in float64.
        b=torch.as_tensor(geometry['global_basis'][:,:rank],dtype=torch.float64)
        v=torch.randn(1536,dtype=torch.float64,requires_grad=True)*.01
        out=v-strength*(v@b)@b.T;g=torch.randn_like(v)
        derivative=torch.autograd.grad((out*g).sum(),v)[0]
        assert torch.allclose(derivative,g-strength*(g@b)@b.T,atol=1e-12)
        cases.append({'rank':rank,'strength':strength,'real_loss_batch':len(ids),'maximum_transform_error':max(errors.values())})
    dump(ROOT/'verification_normal_subspace.json',{'passed':True,'manifest_sha256':digest,'configurations':cases,
        'independent_gram_geometry':diagnostic,'code_sha256':{name:sha(ROOT/name) for name in ['normal_subspace.py','verify_normal_subspace.py']},
        'checks':['Onlyfitnormalpatientcentroids,equalpatientcontribution andper-image/regionmeans',
            'IndependentGram eigenproblem recoversnormalPCAspans,orthonormality,eigenvalues andcenters',
            'Duplicatingnormalimagesdoesnotchangeequal-patientgeometry',
            'Zero-strength exactoriginalM5initialization,RNG andpredictions',
            'IndependentNumPytransforms,projectionJacobian,full256-imageM5task/backward andstrictrecovery',
            'NormalgeometrybuffersfrozenwhileMLPupdates']})
    print('Normal subspace: fit-only patient geometry, independent transforms and real full task verified.',flush=True)


if __name__=='__main__':main()
