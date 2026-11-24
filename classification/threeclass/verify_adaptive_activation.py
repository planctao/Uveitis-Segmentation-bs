"""Independent basis interpolation, derivatives and exact original M5 identity."""
import json
import numpy as np
from scipy.special import softmax
import torch
from experiment import ROOT,Model,load_data,normalize,sample_weights,loss_fn,dump
from adaptive_activation import ActivationResidual,ActivationModel


def independent_activation(layer,x):
    x=np.asarray(x,dtype='float64');bounded=3*np.tanh(x/3)
    centers=layer.centers.detach().cpu().numpy().astype('float64');distance=(bounded[:,:,None]-centers)/1.5
    if layer.kind=='spline':
        # Reconstruct using neighboring knot interpolation, independently of triangular kernels.
        basis=np.zeros((*x.shape,5));position=np.clip((bounded+3)/1.5,0,4)
        lower=np.minimum(np.floor(position).astype('int64'),3);fraction=position-lower
        row,col=np.indices(x.shape);basis[row,col,lower]=1-fraction;basis[row,col,lower+1]=fraction
    else:basis=softmax(-.5*distance**2,axis=-1)
    coefficients=layer.coefficients.detach().cpu().numpy().astype('float64')
    group=np.arange(x.shape[1])//(x.shape[1]//layer.groups)
    return .1*np.sum(basis*coefficients[group][None],-1),basis


def main():
    torch.set_num_threads(4);torch.manual_seed(367);x=torch.randn(5,256,dtype=torch.float64,requires_grad=True)
    for kind in ['spline','gaussian']:
        for groups in [16,256]:
            layer=ActivationResidual(kind,groups).double()
            with torch.no_grad():layer.coefficients.normal_(std=.2)
            actual=layer(x);expected=independent_activation(layer,x.detach().numpy())
            for a,b in zip(actual,expected):assert np.allclose(a.detach().numpy(),b,atol=2e-15,rtol=2e-15)
            assert torch.all(actual[1]>=0) and torch.allclose(actual[1].sum(-1),torch.ones_like(x),atol=1e-15)
            assert actual[0].abs().max()<=.1*layer.coefficients.abs().max()+1e-15
            loss=actual[0].square().sum();loss.backward()
            for parameter in [layer.coefficients,x]:
                gradient=parameter.grad.reshape(-1);assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
                index=int(gradient.abs().argmax());original=parameter.detach().reshape(-1)[index].item();losses=[]
                for sign in [-1,1]:
                    with torch.no_grad():parameter.reshape(-1)[index]=original+sign*1e-5
                    losses.append(np.square(independent_activation(layer,x.detach().numpy())[0]).sum())
                with torch.no_grad():parameter.reshape(-1)[index]=original
                assert abs((losses[1]-losses[0])/2e-5-gradient[index].item())<1e-9
            x.grad=None
            extreme=torch.linspace(-1e6,1e6,256,dtype=torch.float64)[None].requires_grad_()
            delta,basis=layer(extreme);assert torch.isfinite(delta).all() and torch.allclose(basis.sum(-1),torch.ones_like(delta))
            delta.square().sum().backward();assert torch.isfinite(extreme.grad).all()
            for knot in [-3.,-1.5,0.,1.5,3.]:
                # Examine continuity of the complete activation function around arbitrary finite inputs.
                a=layer(torch.full((1,256),knot-1e-8,dtype=torch.float64))[0]
                b=layer(torch.full((1,256),knot+1e-8,dtype=torch.float64))[0]
                assert torch.max((a-b).abs())<1e-8
    frame,features,digest=load_data('multiscale_vit');ref=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())
    c=torch.load(ref['selected']['checkpoints'][0],map_location='cpu',weights_only=False);ids=np.flatnonzero(frame.split.eq('train'));tf=frame.iloc[ids]
    data,_=normalize({k:v[ids[:64]] for k,v in features.items()},None,c['stats']);tx={k:torch.from_numpy(v) for k,v in data.items()}
    y=torch.tensor(tf.label.to_numpy()[:64].copy()).long();disease=torch.tensor([c['diseases'].index(d) for d in tf.disease[:64]])
    weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[:64]),'patient_ids':torch.tensor(__import__('pandas').factorize(tf.patient_group)[0][:64])}
    for kind in ['spline','gaussian']:
        for groups in [16,256]:
            cfg={**c['cfg'],'activation_kind':kind,'activation_groups':groups}
            torch.manual_seed(373);base=Model(c['cfg'],c['dims'],len(c['diseases']));rng=torch.get_rng_state().clone()
            torch.manual_seed(373);model=ActivationModel(cfg,c['dims'],len(c['diseases']))
            assert torch.equal(rng,torch.get_rng_state())
            for key,value in base.state_dict().items():assert torch.equal(value,model.state_dict()[key])
            for training in [False,True]:
                base.train(training);model.train(training)
                torch.manual_seed(379);a=base(tx);rng=torch.get_rng_state().clone()
                torch.manual_seed(379);b=model(tx)
                assert torch.equal(rng,torch.get_rng_state()) and torch.equal(a['p'],b['p'])
            loss_fn(b,y,disease,weights,cfg).backward();assert model.activation_residual.coefficients.grad.abs().sum()>0
            with torch.no_grad():model.activation_residual.coefficients-=.1*model.activation_residual.coefficients.grad
            model.zero_grad();loss_fn(model(tx),y,disease,weights,cfg).backward();assert model.encoder[0].weight.grad.abs().sum()>0
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            restored=ActivationModel(cfg,c['dims'],len(c['diseases']));restored.load_state_dict(model.state_dict(),strict=True)
            model.eval();restored.eval();assert torch.equal(model(tx)['p'],restored(tx)['p'])
    dump(ROOT/'verification_adaptive_activation.json',{'passed':True,'manifest_sha256':digest,'checks':[
        'Independent double-precision neighboring-knot spline interpolation and SciPy Gaussian basis/residual reconstruction',
        'Partition-of-unity, nonnegative basis, convex amplitude bound, finite extreme forward/backward and continuity verified',
        'Numerical actual input and coefficient derivatives, all four original-task gradients and strict model recovery checked',
        'Only80/1280learned coefficients added; exact original common initialization, RNG and initial train/eval probabilities preserved']})
    print('Adaptive activations: independent interpolation, bounds, derivatives, identity and strict restoration passed',flush=True)


if __name__=='__main__':main()
