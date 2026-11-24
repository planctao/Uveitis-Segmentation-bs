"""Independent patient risks, factor dynamics, invariance and task-gradient checks."""
import json
import numpy as np
import torch
from experiment import ROOT, Model, load_data, normalize, sample_weights, loss_fn, dump
from disease_robust import DiseaseRobustWeights


def independent_step(frame, probabilities, previous, eta, shrink, smoothing=.03):
    names=sorted(frame.disease.unique());risks=[];masses=[];parents=[];supports=[]
    labels=frame.label.to_numpy();patient=frame.patient_group.to_numpy()
    logp=np.log(np.maximum(probabilities.astype('float64'),1e-7))
    loss=-(1-smoothing)*logp[np.arange(len(frame)),labels]-smoothing*logp.mean(1)
    # Reconstruct original patient/class image weighting without sample_weights.
    weights=np.zeros(len(frame))
    for parent in range(3):
        ids=np.flatnonzero(labels==parent);patients=np.unique(patient[ids])
        for name in patients:
            rows=ids[patient[ids]==name];weights[rows]=1/3/len(patients)/len(rows)
    for disease in names:
        ids=np.flatnonzero(frame.disease.to_numpy()==disease);mass=weights[ids].sum()
        risks.append(np.dot(loss[ids],weights[ids])/mass);masses.append(mass)
        parents.append(labels[ids[0]]);supports.append(len(np.unique(patient[ids])))
    risks=np.array(risks);mass=np.array(masses);parents=np.array(parents);supports=np.array(supports)
    updated=np.asarray(previous,dtype='float64').copy();factors=np.zeros(len(names))
    for parent in range(3):
        selected=parents==parent;prior=mass[selected]/mass[selected].sum()
        residual=risks[selected]-np.dot(prior,risks[selected])
        updated[selected]=np.clip(updated[selected]+eta*supports[selected]/(supports[selected]+shrink)*residual,-np.log(2),np.log(2))
        raw=np.exp(updated[selected]);factors[selected]=raw/np.dot(raw,prior)
    return risks,updated,factors


def main():
    torch.set_num_threads(4);rng=np.random.default_rng(251)
    import pandas as pd
    frame=pd.DataFrame({'patient_group':['a','a','b','c','d','e','f','g','h'],
        'disease':['a0','a0','b0','a1','b1','b1','n','n','n'],'label':[0,0,0,1,1,1,2,2,2]})
    probabilities=rng.dirichlet(np.ones(3),size=len(frame)).astype('float32')
    for eta,shrink in [(0.,0.),(.05,0.),(.2,0.),(.2,10.)]:
        weights=DiseaseRobustWeights(frame,eta,shrink);state=np.zeros(len(weights.names))
        for step in range(5):
            risks,expected,expected_factors=independent_step(frame,probabilities,state,eta,shrink)
            actual=weights.group_losses(probabilities,frame.label.to_numpy());actual_factors=weights.update(actual)
            assert np.allclose(actual,risks,atol=1e-7,rtol=1e-7)
            assert np.allclose(weights.log_factor,expected,atol=1e-7,rtol=1e-7)
            assert np.allclose(actual_factors,expected_factors,atol=1e-7,rtol=1e-7)
            for parent in range(3):
                mask=weights.parents==parent
                assert np.isclose((actual_factors[mask]*weights.prior[mask]).sum(),1,atol=1e-7)
            assert actual_factors[-1]==1 and np.isfinite(actual_factors).all()
            state=expected
        order=rng.permutation(len(frame));other=DiseaseRobustWeights(frame.iloc[order].reset_index(drop=True),eta,shrink)
        other.update(other.group_losses(probabilities[order],frame.label.to_numpy()[order]))
        original=DiseaseRobustWeights(frame,eta,shrink);original.update(original.group_losses(probabilities,frame.label.to_numpy()))
        assert np.allclose(other.factors(),original.factors(),atol=1e-7)
        duplicate=np.r_[np.arange(len(frame)),[0,1]]
        repeated=DiseaseRobustWeights(frame.iloc[duplicate].reset_index(drop=True),eta,shrink)
        repeated.update(repeated.group_losses(probabilities[duplicate],frame.label.to_numpy()[duplicate]))
        assert np.allclose(repeated.factors(),original.factors(),atol=1e-7)
    real,features,digest=load_data('multiscale_vit');indices=np.flatnonzero(real.split.eq('train'));tf=real.iloc[indices]
    reference=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())['selected']
    c=torch.load(reference['checkpoints'][0],map_location='cpu',weights_only=False)
    data,_=normalize({k:v[indices[:64]] for k,v in features.items()},None,c['stats']);tx={k:torch.from_numpy(v) for k,v in data.items()}
    model=Model(c['cfg'],c['dims'],len(c['diseases']));model.load_state_dict(c['state_dict'],strict=True)
    y=torch.tensor(tf.label.to_numpy()[:64].copy()).long();disease=torch.tensor([c['diseases'].index(d) for d in tf.disease[:64]])
    wt={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[:64]),'patient_ids':torch.tensor(pd.factorize(tf.patient_group)[0][:64])}
    robust=DiseaseRobustWeights(tf,0.);assert np.array_equal(robust.factors(),np.ones(len(robust.names),dtype='float32'))
    model.train();out=model(tx);original=loss_fn(out,y,disease,wt,c['cfg'])
    updated={**wt,'leaf':wt['leaf']*torch.from_numpy(robust.factors()[robust.targets[:64]])}
    assert torch.equal(original,loss_fn(out,y,disease,updated,c['cfg']))
    original.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    dump(ROOT/'verification_disease_robust.json',{'passed':True,'manifest_sha256':digest,'checks':[
        'Independent per-patient/class image masses and per-disease risk moments reproduced without sample_weights',
        'Every exponent update, within-parent normalization and support shrink reproduced independently across repeated steps',
        'Patient-image duplication and order invariance; class mass retained; normal singleton factor remains one',
        'Zero-update arm exactly preserves original complete task loss and finite model gradients; inference architecture unchanged']})
    print('Within-parent disease robust risks, factor dynamics, invariances and zero-control task verified',flush=True)


if __name__=='__main__':main()
