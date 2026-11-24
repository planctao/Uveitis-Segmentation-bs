"""Check diversity geometry, numerical gradients and unchanged classifier outputs."""
import json
import numpy as np
import torch
from experiment import ROOT,Model,load_data,normalize,loss_fn,dump
from attention_diversity import attention_overlap_penalty


def main():
    torch.set_num_threads(4);torch.manual_seed(20261124)
    identical=torch.tensor([[[.2,.2],[.3,.3],[.5,.5]]],dtype=torch.float64)
    distinct=torch.tensor([[[1.,0.],[0.,1.],[0.,0.]]],dtype=torch.float64)
    for margin in [.8,.9]:
        assert torch.allclose(attention_overlap_penalty(identical,margin),torch.tensor([(1-margin)**2],dtype=torch.float64))
        assert not attention_overlap_penalty(distinct,margin).count_nonzero()
    scores=torch.randn(2,7,3,dtype=torch.float64,requires_grad=True)*.2
    assert torch.autograd.gradcheck(lambda z:attention_overlap_penalty(z.softmax(1),.8),(scores,),eps=1e-6,atol=1e-5)
    attention=scores.softmax(1)
    expected=[]
    for sample in attention.detach().numpy():
        pairs=[]
        for i in range(3):
            for j in range(i+1,3):
                cosine=sample[:,i]@sample[:,j]/(np.linalg.norm(sample[:,i])*np.linalg.norm(sample[:,j]))
                pairs.append(max(cosine-.8,0)**2)
        expected.append(np.mean(pairs))
    assert np.allclose(attention_overlap_penalty(attention,.8).detach().numpy(),expected,atol=1e-12)
    assert torch.allclose(attention_overlap_penalty(attention,.8),attention_overlap_penalty(attention[:,[4,1,5,3,0,6,2]],.8))
    ref=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())['selected']
    c=torch.load(ref['checkpoints'][0],map_location='cpu',weights_only=False)
    frame,features,digest=load_data('multiscale_vit');indices=np.flatnonzero(frame.split.eq('val'))[:12]
    x,_=normalize({k:v[indices] for k,v in features.items()},None,c['stats']);inputs={k:torch.from_numpy(v) for k,v in x.items()}
    cfg={**c['cfg'],'attention_diversity_weight':.5,'attention_diversity_margin':.8}
    torch.manual_seed(93);base=Model(c['cfg'],c['dims'],len(c['diseases'])).eval()
    torch.manual_seed(93);diverse=Model(cfg,c['dims'],len(c['diseases'])).eval()
    assert set(base.state_dict())==set(diverse.state_dict())
    for key,value in base.state_dict().items():assert torch.equal(diverse.state_dict()[key],value)
    diverse.load_state_dict(c['state_dict'],strict=True);base.load_state_dict(c['state_dict'],strict=True)
    a,b=base(inputs),diverse(inputs);assert torch.allclose(a['p'],b['p'],atol=1e-7,rtol=1e-7)
    y=torch.tensor(frame.iloc[indices].label.to_numpy().copy()).long()
    disease=torch.tensor([c['diseases'].index(name) for name in frame.iloc[indices].disease])
    weights={'leaf':torch.linspace(.5,1.5,len(indices)),'patient_ids':torch.arange(len(indices))}
    original=loss_fn(a,y,disease,weights,c['cfg']);augmented=loss_fn(b,y,disease,weights,cfg)
    expected=.5*(attention_overlap_penalty(b['region_attention'],.8)*weights['leaf']).mean()
    assert torch.allclose(augmented-original,expected,atol=2e-6) and expected>0
    gradient=torch.autograd.grad(expected,list(diverse.region_gate.parameters()))
    assert all(torch.isfinite(g).all() for g in gradient) and sum(g.abs().sum() for g in gradient)>0
    dump(ROOT/'verification_attention_diversity.json',{'passed':True,'manifest_sha256':digest,
        'checks':['Identical attention has exact squared excess overlap; disjoint attention has zero penalty',
            'Doubleprecision numerical gradient check with3heads passed',
            'Independent NumPy pairwise cosine geometry and region permutation invariance',
            'No newparameters or randomdraws; allcommon tensors and predictions unchanged',
            'Full patientweighted loss difference equals declared regularizer; finite nonzero attentiongategradients',
            'Original protected checkpoint loads strictly with diversity enabled']})
    print('Attention diversity geometry, gradients and unchanged classifier checks passed',flush=True)


if __name__=='__main__':main()
