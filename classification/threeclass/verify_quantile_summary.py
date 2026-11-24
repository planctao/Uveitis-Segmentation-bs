"""Independent masked NumPy quantiles, task gradients and baseline identity."""
import json
import numpy as np
from scipy.special import ndtr
import torch
from experiment import ROOT,Model,load_data,normalize,sample_weights,loss_fn,dump
from quantile_summary import QuantileSummary,QuantileModel,regional_quantiles


def independent_quantile(layer,tokens,mask=None):
    state={k:v.detach().cpu().numpy().astype('float64') for k,v in layer.state_dict().items()}
    x=np.asarray(tokens,dtype='float64')
    z=(x-x.mean(-1,keepdims=True))/np.sqrt(x.var(-1,keepdims=True)+1e-5)
    z=z*state['norm.weight']+state['norm.bias']
    values=z@state['down.weight'].T+state['down.bias'];values*=ndtr(values)
    valid=np.ones(x.shape[:2],dtype=bool) if mask is None else np.asarray(mask)>.5
    summaries=[]
    for row in range(len(x)):
        select=valid[row] if valid[row].any() else np.ones(x.shape[1],dtype=bool)
        if layer.kind=='quantile':s=np.quantile(values[row,select],[.25,.5,.75],axis=0,method='linear')
        else:s=np.repeat(values[row,select].mean(0)[None],3,axis=0)
        summaries.append(s)
    summary=np.stack(summaries)
    return .1*summary.reshape(len(x),-1)@state['readout.weight'].T,summary,values


def main():
    torch.set_num_threads(4);torch.manual_seed(383);x=torch.randn(3,9,128,dtype=torch.float64)
    mask=torch.tensor([[1]*9,[1,0,0,0,0,0,0,0,0],[0]*9]);permutation=torch.tensor([7,4,1,8,5,2,6,3,0])
    for rank in [16,32]:
        counts=[]
        for kind in ['mean','quantile']:
            torch.manual_seed(389);layer=QuantileSummary(rank,kind).double();counts.append(sum(p.numel() for p in layer.parameters()))
            with torch.no_grad():layer.readout.weight.normal_(std=.04)
            actual=layer(x,mask);expected=independent_quantile(layer,x.numpy(),mask.numpy())
            for a,b in zip(actual[:3],expected):assert np.allclose(a.detach().numpy(),b,atol=1e-12,rtol=1e-12)
            shuffled=layer(x[:,permutation],mask[:,permutation]);assert torch.allclose(shuffled[0],actual[0],atol=1e-12)
            assert torch.allclose(shuffled[1],actual[1],atol=1e-12)
            single=layer(x[:,:1]);assert torch.allclose(single[1][:,0],single[1][:,1]) and torch.allclose(single[1][:,1],single[1][:,2])
            changed=x.clone();changed[1,1:]=1e4;assert torch.equal(layer(changed,mask)[0][1],actual[0][1])
            loss=actual[0].square().sum();loss.backward();p=layer.down.weight;gradient=p.grad.flatten();index=int(gradient.abs().argmax());original=p.detach().flatten()[index].item();losses=[]
            for sign in [-1,1]:
                with torch.no_grad():p.flatten()[index]=original+sign*1e-5
                losses.append(np.square(independent_quantile(layer,x.numpy(),mask.numpy())[0]).sum())
            with torch.no_grad():p.flatten()[index]=original
            assert abs((losses[1]-losses[0])/2e-5-gradient[index].item())<2e-8
        assert counts[0]==counts[1]
    # Same mean, different regional distribution: the summary distinguishes the inputs.
    a=torch.tensor([-4.,-3.,-2.,-1.,0.,1.,2.,3.,4.]).reshape(1,9,1)
    b=torch.tensor([-8.,0.,0.,0.,0.,0.,0.,0.,8.]).reshape(1,9,1)
    assert a.mean()==b.mean() and not torch.equal(regional_quantiles(a)[0],regional_quantiles(b)[0])
    tied=torch.zeros(2,9,3,requires_grad=True);q,_=regional_quantiles(tied);q.sum().backward();assert torch.isfinite(tied.grad).all()
    frame,features,digest=load_data('multiscale_vit');ref=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())
    c=torch.load(ref['selected']['checkpoints'][0],map_location='cpu',weights_only=False);ids=np.flatnonzero(frame.split.eq('train'));tf=frame.iloc[ids]
    data,_=normalize({k:v[ids[:64]] for k,v in features.items()},None,c['stats']);tx={k:torch.from_numpy(v) for k,v in data.items()}
    y=torch.tensor(tf.label.to_numpy()[:64].copy()).long();disease=torch.tensor([c['diseases'].index(d) for d in tf.disease[:64]])
    weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[:64]),'patient_ids':torch.tensor(__import__('pandas').factorize(tf.patient_group)[0][:64])}
    for rank in [16,32]:
        for kind in ['mean','quantile']:
            cfg={**c['cfg'],'quantile_kind':kind,'quantile_rank':rank}
            torch.manual_seed(397);base=Model(c['cfg'],c['dims'],len(c['diseases']));rng=torch.get_rng_state().clone()
            torch.manual_seed(397);model=QuantileModel(cfg,c['dims'],len(c['diseases']))
            assert torch.equal(rng,torch.get_rng_state())
            for key,value in base.state_dict().items():assert torch.equal(value,model.state_dict()[key])
            for training in [False,True]:
                base.train(training);model.train(training)
                torch.manual_seed(401);aa=base(tx);rng=torch.get_rng_state().clone()
                torch.manual_seed(401);bb=model(tx)
                assert torch.equal(rng,torch.get_rng_state()) and torch.equal(aa['p'],bb['p'])
            loss_fn(bb,y,disease,weights,cfg).backward();assert model.quantile_summary.readout.weight.grad.abs().sum()>0
            with torch.no_grad():model.quantile_summary.readout.weight-=.1*model.quantile_summary.readout.weight.grad
            model.zero_grad();loss_fn(model(tx),y,disease,weights,cfg).backward();assert model.quantile_summary.down.weight.grad.abs().sum()>0
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            restored=QuantileModel(cfg,c['dims'],len(c['diseases']));restored.load_state_dict(model.state_dict(),strict=True)
            model.eval();restored.eval();assert torch.equal(model(tx)['p'],restored(tx)['p'])
    dump(ROOT/'verification_quantile_summary.json',{'passed':True,'manifest_sha256':digest,'checks':[
        'Independent masked NumPy linear quantiles, mean summaries, full projected values and residual formula verified',
        'Matched mean/quantile parameter counts, numerical actual down-projection derivative and same-mean distribution contrast checked',
        'Region permutation invariance, invalid regions exclusion, singleton/allmasked fallback and tied-value finite gradients checked',
        'All four original task gradients, exact common M5 initialization/RNG/train/evalidentity and strict full restoration verified']})
    print('Quantile summaries: independent percentiles, matched controls, boundaries, gradients and M5 identity passed',flush=True)


if __name__=='__main__':main()
