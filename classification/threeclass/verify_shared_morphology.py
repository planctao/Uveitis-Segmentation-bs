"""Validate shared-family mixtures, rare-disease prior shrinkage and integration."""
import json
import numpy as np
from scipy.special import logsumexp,softmax
import torch
from experiment import ROOT,Model,load_data,normalize,loss_fn,dump
from shared_morphology import SharedMorphology


def main():
    torch.set_num_threads(4);torch.manual_seed(20261124)
    parents=[0,0,1,1,2];counts=[2,50,2,50,111]
    layer=SharedMorphology(2,parents,counts,2).double()
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[[1.,0.],[-1.,0.]]]).repeat(3,1,1))
        layer.prior_logits.copy_(torch.tensor([[2.,-2.]]).repeat(5,1))
    x=torch.tensor([[-2.,0.],[0.,0.],[2.,0.]],dtype=torch.float64,requires_grad=True)
    evidence,components,log_priors=layer(x)
    expected_priors=softmax(np.array([[2.,-2.]])*np.sqrt(np.array(counts)/(np.array(counts)+10))[:,None],axis=1)
    assert np.allclose(log_priors.exp().detach().numpy(),expected_priors,atol=1e-7)
    assert expected_priors[0].max()<expected_priors[1].max()
    assert torch.equal(components[:,0],components[:,1]) and torch.equal(components[:,2],components[:,3])
    expected=logsumexp(components.detach().numpy()+np.log(expected_priors)[None],axis=-1)
    assert np.allclose(evidence.detach().numpy(),expected,atol=1e-7)
    assert torch.autograd.gradcheck(lambda z:layer(z)[0],(x,),atol=1e-5)
    original=evidence.detach().clone()
    with torch.no_grad():
        layer.weight.copy_(layer.weight[:,[1,0]]);layer.bias.copy_(layer.bias[:,[1,0]])
        layer.prior_logits.copy_(layer.prior_logits[:,[1,0]])
    assert torch.allclose(layer(x)[0],original,atol=1e-12)
    one=SharedMorphology(2,parents,counts,1).double();assert not one(x)[0].count_nonzero()
    ref=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())['selected']
    c=torch.load(ref['checkpoints'][0],map_location='cpu',weights_only=False)
    frame,features,digest=load_data('multiscale_vit');fit=frame[frame.split.eq('train')]
    count=fit.groupby('disease').patient_group.nunique();dims={**c['dims'],'disease_patient_counts':[int(count[name]) for name in c['diseases']]}
    indices=np.flatnonzero(frame.split.eq('val'))[:16];data,_=normalize({k:v[indices] for k,v in features.items()},None,c['stats'])
    inputs={k:torch.from_numpy(v) for k,v in data.items()};proof=[]
    for k in [2,4]:
        cfg={**c['cfg'],'taxonomy_shared_morphology':k,'shared_prior_shrink':10.,'shared_morphology_init_std':.005,'shared_morphology_penalty':.01}
        torch.manual_seed(93);base=Model(c['cfg'],dims,len(c['diseases'])).eval()
        torch.manual_seed(93);shared=Model(cfg,dims,len(c['diseases'])).eval()
        for key,value in base.state_dict().items():assert torch.equal(shared.state_dict()[key],value)
        base.load_state_dict(c['state_dict'],strict=True);missing,unexpected=shared.load_state_dict(c['state_dict'],strict=False)
        assert all(name.startswith('shared_morphology.') for name in missing) and not unexpected
        a,b=base(inputs),shared(inputs);difference=float((a['p']-b['p']).abs().max().detach());assert difference<.005
        y=torch.tensor(frame.iloc[indices].label.to_numpy().copy()).long()
        disease=torch.tensor([c['diseases'].index(name) for name in frame.iloc[indices].disease])
        weights={'leaf':torch.ones(len(indices)),'patient_ids':torch.arange(len(indices))}
        loss=loss_fn(b,y,disease,weights,cfg);unregularized=loss_fn(b,y,disease,weights,{**cfg,'shared_morphology_penalty':0.})
        assert torch.allclose(loss-unregularized,.01*b['morphology_regularizer'],atol=2e-6)
        loss.backward();assert shared.shared_morphology.weight.grad.abs().sum()>0 and shared.shared_morphology.prior_logits.grad.abs().sum()>0
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in shared.parameters())
        restored=Model(cfg,dims,len(c['diseases'])).eval();restored.load_state_dict(shared.state_dict(),strict=True)
        assert torch.allclose(restored(inputs)['p'],b['p'],atol=1e-7)
        proof.append({'components':k,'initial_probability_max_abs_change':difference,
            'initial_class_decisions_changed_in_checked16':int((a['p'].argmax(1)!=b['p'].argmax(1)).sum()),
            'new_parameters':sum(p.numel() for p in shared.shared_morphology.parameters())})
    dump(ROOT/'verification_shared_morphology.json',{'passed':True,'manifest_sha256':digest,'initialization_diagnostics':proof,
        'checks':['Independent SciPy prior-softmax and logmixture formula; rare disease priors closer to uniform',
            'Same-family diseases use exactly shared componentvalues; componentpermutation invariant',
            'Doubleprecision numericalgradientcheck and singlecomponentzeroresidual',
            'Common model tensors initializedidentically; small initial predictionchanges explicitly measured',
            'Full loss uses exactly declared penalty; finite nonzero dictionary and diseaseprior gradients',
            'Strict saved/restored full model probabilities match']})
    print(json.dumps({'verification':'passed','initialization':proof}),flush=True)


if __name__=='__main__':main()
