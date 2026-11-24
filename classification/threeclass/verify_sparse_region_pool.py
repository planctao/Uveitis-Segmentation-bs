"""Independent scalar-root probabilities and numerical sparse-pooling checks."""
import json
import numpy as np
from scipy.optimize import brentq
from scipy.special import softmax
import torch
from experiment import ROOT, Model, load_data, normalize, loss_fn, sample_weights, dump
from sparse_region_pool import region_distribution, SparseRegionModel


def independent_attention(scores, kind, temperature, mask=None):
    scores = np.asarray(scores, dtype='float64'); n, count, heads = scores.shape
    valid = np.ones((n, count), dtype=bool) if mask is None else np.asarray(mask) > .5
    result = np.zeros_like(scores)
    for row in range(n):
        allowed = valid[row].copy()
        if not allowed.any(): allowed[:] = True
        for head in range(heads):
            x = scores[row, allowed, head] / temperature
            if kind == 'softmax': p = softmax(x)
            else:
                x = x - x.max()
                if kind == 'entmax15': x = x / 2
                exponent = 2 if kind == 'entmax15' else 1
                tau = brentq(lambda v: np.maximum(x - v, 0).__pow__(exponent).sum() - 1,
                             float(x.min()) - 2, float(x.max()), xtol=1e-14)
                p = np.maximum(x - tau, 0) ** exponent; p /= p.sum()
            result[row, allowed, head] = p
    return result


def main():
    torch.set_num_threads(4); torch.manual_seed(211)
    scores = torch.randn(4, 9, 2, dtype=torch.float64) * 2
    masks = [None, np.array([[1]*9, [1,0,0,0,0,0,0,0,0], [0]*9, [1,1,0,0,1,0,0,0,0]])]
    coefficient = np.random.default_rng(211).normal(size=(4,9,2))
    for kind in ['softmax','entmax15','sparsemax']:
        for temperature in [.5,1.,2.]:
            for mask in masks:
                sm=scores.clone().requires_grad_();tm=None if mask is None else torch.from_numpy(mask)
                actual=region_distribution(sm,kind,temperature,tm)
                expected=independent_attention(scores.numpy(),kind,temperature,mask)
                assert np.allclose(actual.detach().numpy(),expected,atol=1e-11,rtol=1e-11)
                loss=(actual*torch.from_numpy(coefficient)).sum();loss.backward()
                assert torch.isfinite(sm.grad).all()
                for index in [0,13,31]:
                    losses=[]
                    for sign in [-1,1]:
                        modified=scores.numpy().copy();modified.reshape(-1)[index]+=sign*1e-5
                        losses.append((independent_attention(modified,kind,temperature,mask)*coefficient).sum())
                    assert abs((losses[1]-losses[0])/2e-5-sm.grad.reshape(-1)[index].item())<1e-7
                shift=torch.randn(4,1,2,dtype=torch.float64)*100
                assert torch.allclose(actual,region_distribution(scores+shift,kind,temperature,tm),atol=1e-12)
                order=torch.tensor([8,6,4,2,0,7,5,3,1])
                assert torch.allclose(actual[:,order],region_distribution(scores[:,order],kind,temperature,None if tm is None else tm[:,order]),atol=1e-12)
        extreme=torch.tensor([[[1e4],[-1e4]]],dtype=torch.float64,requires_grad=True)
        p=region_distribution(extreme,kind);p.square().sum().backward()
        assert torch.isfinite(extreme.grad).all() and torch.isfinite(p).all()
        equal=region_distribution(torch.zeros(2,9,2,dtype=torch.float64),kind)
        assert torch.allclose(equal,torch.full_like(equal,1/9),atol=1e-12)
    assert (region_distribution(scores,'sparsemax')==0).any() and (region_distribution(scores,'entmax15')==0).any()
    for values in [[0,2],[0,2,4],[0,0,2,2],[-2,-1,0,1,2],[-10,0,1,2,10],[0]*9]:
        for device in ['cpu','cuda']:
            x=torch.tensor(values,dtype=torch.float64,device=device).reshape(1,-1,1).requires_grad_()
            p=region_distribution(x,'entmax15')
            (p*torch.arange(len(values),device=device).reshape(1,-1,1)).sum().backward()
            assert torch.isfinite(x.grad).all()
            assert np.allclose(p.detach().cpu().numpy(),independent_attention(x.detach().cpu().numpy(),'entmax15',1.),atol=1e-11)

    frame,features,digest=load_data('multiscale_vit');train=np.flatnonzero(frame.split.eq('train'));tf=frame.iloc[train]
    ref=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())['selected']
    c=torch.load(ref['checkpoints'][0],map_location='cpu',weights_only=False)
    data,_=normalize({k:v[train[:64]] for k,v in features.items()},None,c['stats']);tx={k:torch.from_numpy(v) for k,v in data.items()}
    y=torch.tensor(tf.label.to_numpy()[:64].copy()).long();disease=torch.tensor([c['diseases'].index(d) for d in tf.disease[:64]])
    weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[:64]),'patient_ids':torch.tensor(__import__('pandas').factorize(tf.patient_group)[0][:64])}
    for kind,temperature in [('softmax',1.),('softmax',.5),('softmax',2.),('entmax15',1.),('sparsemax',1.)]:
        cfg={**c['cfg'],'region_pool_kind':kind,'region_pool_temperature':temperature}
        torch.manual_seed(213);base=Model(c['cfg'],c['dims'],len(c['diseases']));rng=torch.get_rng_state().clone()
        torch.manual_seed(213);model=SparseRegionModel(cfg,c['dims'],len(c['diseases']))
        assert torch.equal(rng,torch.get_rng_state()) and set(base.state_dict())==set(model.state_dict())
        for key,value in base.state_dict().items():assert torch.equal(value,model.state_dict()[key])
        for training in [False,True]:
            base.train(training);model.train(training)
            torch.manual_seed(215);a=base(tx);rng=torch.get_rng_state().clone()
            torch.manual_seed(215);b=model(tx);assert torch.equal(rng,torch.get_rng_state())
            if kind=='softmax' and temperature==1.:assert torch.equal(a['p'],b['p'])
        loss_fn(b,y,disease,weights,cfg).backward()
        assert model.region_gate[0].weight.grad.abs().sum()>0 and model.encoder[0].weight.grad.abs().sum()>0
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        restored=SparseRegionModel(cfg,c['dims'],len(c['diseases']));restored.load_state_dict(model.state_dict(),strict=True)
        model.eval();restored.eval();assert torch.equal(model(tx)['p'],restored(tx)['p'])
    dump(ROOT/'verification_sparse_region_pool.json',{'passed':True,'manifest_sha256':digest,'checks':[
        'Entmax1.5/sparsemax distributions independently solved by scalar SciPy roots, softmax by independent SciPy function',
        'Selected numerical score derivatives, shift/permutation invariance, masked/single-valid/allmasked/equal/extreme boundaries verified',
        'All parameter tensors and random streams identical to M5, no new parameters; ordinary softmax temperature1 exactly restores original model',
        'Actual task gradients finite and reach regional scores/encoder; strict model recovery exact']})
    print('Sparse region pool formulas, gradients, boundaries, parameter/RNG identity and baseline recovery verified',flush=True)


if __name__=='__main__':main()
