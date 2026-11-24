"""Independent mixture algebra, patient shrinkage, gradients and checkpoint checks."""
import json
import numpy as np
from scipy.special import logsumexp
import torch
from experiment import ROOT,Model,load_data,normalize,loss_fn,dump
from subcluster_head import SubclusterEvidence


def main():
    torch.set_num_threads(4);torch.manual_seed(20261124)
    layer=SubclusterEvidence(2,[2,50],2,10.,.005).double()
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[[1.,0.],[-1.,0.]],[[1.,0.],[-1.,0.]]]))
    x=torch.tensor([[-2.,0.],[0.,0.],[2.,0.]],dtype=torch.float64,requires_grad=True)
    evidence,components=layer(x)
    expected=np.log(np.cosh(np.array([[-2.],[0.],[2.]])*np.sqrt(np.array([[2/12,50/60]]))))
    assert np.allclose(evidence.detach().numpy(),expected,atol=1e-7)
    assert torch.allclose(evidence[0],evidence[2]) and torch.allclose(evidence[1],torch.zeros(2,dtype=torch.float64))
    assert torch.autograd.gradcheck(lambda z:layer(z)[0],(x,),atol=1e-5)
    one=SubclusterEvidence(2,[2,50],1).double()
    assert not one(x)[0].count_nonzero()
    random=SubclusterEvidence(4,[2,10,100],4).double()
    z=torch.randn(7,4,dtype=torch.float64);e,c=random(z);w,b=random.centered_parameters()
    expected=logsumexp(np.einsum('bh,dkh->bdk',z.numpy(),w.detach().numpy())+b.detach().numpy()[None],axis=-1)-np.log(4)
    assert np.allclose(e.detach().numpy(),expected,atol=1e-12)
    assert torch.allclose(c.mean(-1),torch.zeros(7,3,dtype=torch.float64),atol=1e-12) and e.min()>-1e-12
    original=e.detach().clone()
    with torch.no_grad():
        random.weight.add_(torch.randn(3,1,4));random.bias.add_(torch.randn(3,1))
    assert torch.allclose(random(z)[0],original,atol=1e-12)
    with torch.no_grad():
        random.weight.copy_(random.weight[:,[2,0,3,1]]);random.bias.copy_(random.bias[:,[2,0,3,1]])
    assert torch.allclose(random(z)[0],original,atol=1e-12)
    ref=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())['selected']
    c=torch.load(ref['checkpoints'][0],map_location='cpu',weights_only=False)
    frame,features,digest=load_data('multiscale_vit');fit=frame[frame.split.eq('train')]
    counts=fit.groupby('disease').patient_group.nunique();dims={**c['dims'],'disease_patient_counts':[int(counts[name]) for name in c['diseases']]}
    indices=np.flatnonzero(frame.split.eq('val'))[:16]
    x,_=normalize({k:v[indices] for k,v in features.items()},None,c['stats']);inputs={k:torch.from_numpy(v) for k,v in x.items()}
    proof=[]
    for components in [2,4]:
        cfg={**c['cfg'],'taxonomy_subclusters':components,'subcluster_shrink':10.,'subcluster_init_std':.005,'subcluster_penalty':.01}
        torch.manual_seed(93);base=Model(c['cfg'],dims,len(c['diseases'])).eval()
        torch.manual_seed(93);mixture=Model(cfg,dims,len(c['diseases'])).eval()
        for key,value in base.state_dict().items():assert torch.equal(mixture.state_dict()[key],value)
        # Small random centered components break the exact symmetry; initialization is intentionally only approximately neutral.
        base.load_state_dict(c['state_dict'],strict=True)
        missing,unexpected=mixture.load_state_dict(c['state_dict'],strict=False)
        assert set(missing)=={'disease_clusters.weight','disease_clusters.bias','disease_clusters.support_scale'} and not unexpected
        a,b=base(inputs),mixture(inputs)
        difference=float((a['p']-b['p']).abs().max().detach());assert difference<.005
        assert torch.allclose(mixture.disease_clusters.support_scale,torch.tensor(dims['disease_patient_counts']).div(torch.tensor(dims['disease_patient_counts'])+10).sqrt())
        y=torch.tensor(frame.iloc[indices].label.to_numpy().copy()).long()
        disease=torch.tensor([c['diseases'].index(name) for name in frame.iloc[indices].disease])
        weights={'leaf':torch.ones(len(indices)),'patient_ids':torch.arange(len(indices))}
        loss=loss_fn(b,y,disease,weights,cfg);assert torch.isfinite(loss)
        loss.backward()
        assert mixture.disease_clusters.weight.grad.abs().sum()>0
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in mixture.parameters())
        fresh=Model(cfg,dims,len(c['diseases'])).eval();fresh.load_state_dict(mixture.state_dict(),strict=True)
        assert torch.allclose(fresh(inputs)['p'],b['p'],atol=1e-7)
        proof.append({'components':components,'initial_probability_max_abs_change':difference,
            'initial_class_decisions_changed_in_checked16':int((a['p'].argmax(1)!=b['p'].argmax(1)).sum())})
    dump(ROOT/'verification_subcluster_head.json',{'passed':True,'manifest_sha256':digest,'initialization_diagnostics':proof,
        'checks':['Exact symmetric mixture equals logcosh with analytically known patientcounts shrinkage',
            'Singlecomponent reduces to zero residual; doubleprecision input gradient check passed',
            'Independent SciPy logmeanexp, zero componentmeans and nonnegative evidence verified',
            'Shared componentoffsets cancel; componentpermutations preserve evidence',
            'AllcommonM5tensors initializedidentically; small randomresiduals intentionally break symmetry',
            'Actualfittingpatientcounts set shrinkage; full loss gives finite nonzero mixturegradients',
            'Strict full checkpoint restore preserves predictions']})
    print(json.dumps({'verification':'passed','initialization':proof}),flush=True)


if __name__=='__main__':main()
