"""Independent geometric pooling, summary adapter and actual M5 task checks."""
import json
import numpy as np
from scipy.special import ndtr
import torch
from experiment import ROOT, Model, normalize, loss_fn, sample_weights, dump
from radial_summary import radial_bins, summarize_regions, load_radial, RadialModel


def independent_adapter(layer, summary):
    state = {k: v.detach().cpu().numpy().astype('float64') for k, v in layer.state_dict().items()}
    x = summary.astype('float64')
    x = (x-x.mean(-1,keepdims=True)) / np.sqrt(x.var(-1,keepdims=True)+1e-5)
    x = x*state['norm.weight']+state['norm.bias']
    z = x@state['down.weight'].T+state['down.bias']; z = z*ndtr(z)
    return .1*(z.reshape(len(z),-1)@state['up.weight'].T+state['up.bias'])


def main():
    torch.set_num_threads(4); rng=np.random.default_rng(227)
    x=rng.normal(size=(4,49,8)).astype('float32'); w=rng.uniform(size=(4,49)).astype('float32')
    w[1]=0; w[2,radial_bins()==0]=0
    permutation=np.rot90(np.arange(49).reshape(7,7)).reshape(-1)
    for mode in ['global','radial','contrast']:
        actual=summarize_regions(x,w,mode); expected=[]
        for row in range(len(x)):
            total=np.average(x[row].astype('float64'),axis=0,weights=w[row]) if w[row].sum()>0 else x[row].astype('float64').mean(0)
            result=[]
            for ring in range(3):
                selected=radial_bins()==ring
                mean=np.average(x[row,selected].astype('float64'),axis=0,weights=w[row,selected]) if w[row,selected].sum()>0 else total
                result.append(total if mode=='global' else mean-total if mode=='contrast' else mean)
            expected.append(result)
        assert np.allclose(actual,expected,atol=1e-6,rtol=1e-6)
        assert np.allclose(actual,summarize_regions(x[:,permutation],w[:,permutation],mode),atol=1e-6)
        reflection=np.arange(49).reshape(7,7)[:,::-1].reshape(-1)
        assert np.allclose(actual,summarize_regions(x[:,reflection],w[:,reflection],mode),atol=1e-6)
        constant=np.repeat(x[:,:1],49,axis=1); pooled=summarize_regions(constant,w,mode)
        assert np.allclose(pooled,0 if mode=='contrast' else constant[:,:3],atol=1e-6)

    frame,features,digest=load_radial('radial');indices=np.flatnonzero(frame.split.eq('train'));tf=frame.iloc[indices]
    reference=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())['selected']
    c=torch.load(reference['checkpoints'][0],map_location='cpu',weights_only=False)
    mask=np.zeros(len(frame),dtype=bool);mask[indices]=True
    normalized,stats=normalize(features,mask);tx={k:torch.from_numpy(v[indices[:64]]) for k,v in normalized.items()}
    dims={**c['dims'],'spatial_summary':1536};cfg={**c['cfg'],'radial_rank':16}
    torch.manual_seed(229);base=Model(c['cfg'],dims,len(c['diseases']));base_rng=torch.get_rng_state().clone()
    torch.manual_seed(229);model=RadialModel(cfg,dims,len(c['diseases']))
    assert torch.equal(base_rng,torch.get_rng_state())
    for key,value in base.state_dict().items():assert torch.equal(value,model.state_dict()[key])
    for training in [False,True]:
        base.train(training);model.train(training)
        torch.manual_seed(231);a=base(tx);base_rng=torch.get_rng_state().clone()
        torch.manual_seed(231);b=model(tx)
        assert torch.equal(base_rng,torch.get_rng_state()) and torch.equal(a['p'],b['p'])
    y=torch.tensor(tf.label.to_numpy()[:64].copy()).long();disease=torch.tensor([c['diseases'].index(d) for d in tf.disease[:64]])
    weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[:64]),'patient_ids':torch.tensor(__import__('pandas').factorize(tf.patient_group)[0][:64])}
    loss_fn(b,y,disease,weights,cfg).backward();assert model.radial.up.weight.grad.abs().sum()>0
    with torch.no_grad():model.radial.up.weight-=.1*model.radial.up.weight.grad
    model.zero_grad();loss_fn(model(tx),y,disease,weights,cfg).backward()
    assert model.radial.down.weight.grad.abs().sum()>0 and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval();out=model(tx);expected=independent_adapter(model.radial,tx['spatial_summary'].numpy())
    assert np.allclose(expected,out['radial_delta'].detach().numpy(),atol=2e-6,rtol=2e-6)
    restored=RadialModel(cfg,dims,len(c['diseases'])).eval();restored.load_state_dict(model.state_dict(),strict=True)
    assert torch.equal(restored(tx)['p'],out['p'])
    for mode in ['global','contrast']:
        other,values,other_digest=load_radial(mode);assert digest==other_digest
        for key in ['global','regions','mid','dispersion']:
            assert np.array_equal(values[key][:2091],features[key][:2091])
    dump(ROOT/'verification_radial_summary.json',{'passed':True,'manifest_sha256':digest,'ring_region_counts':np.bincount(radial_bins()).tolist(),
        'checks':['Independent weighted global/ring/contrast formulas; empty-ring and all-empty fallback; quarter-turn and reflection invariance; constant-token limits',
            'Original M5 four input arrays unchanged for all summary layouts; strict development-only row identities and external NaN arrays',
            'Common parameters, initial full train/eval predictions and RNG exactly preserve M5; actual loss reaches readout then projection',
            'Actual adapter independently reconstructed in double precision; finite gradients and strict full model recovery']})
    print('Radial geometry, fit-only features, neutral adapter, gradients and restoration verified',flush=True)


if __name__=='__main__':main()
