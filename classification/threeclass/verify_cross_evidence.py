"""Independent global/local interaction formula and complete M5 identity checks."""
import json
import math
import numpy as np
from scipy.special import ndtr
import torch
from experiment import ROOT,Model,load_data,normalize,sample_weights,loss_fn,dump
from cross_evidence import CrossEvidence,CrossEvidenceModel


def independent_cross(layer,fused):
    state={k:v.detach().cpu().numpy().astype('float64') for k,v in layer.state_dict().items()}
    def norm(x,prefix):
        z=(x-x.mean(-1,keepdims=True))/np.sqrt(x.var(-1,keepdims=True)+1e-5)
        return z*state[prefix+'.weight']+state[prefix+'.bias']
    x=np.asarray(fused,dtype='float64')
    a=norm(x[:,:256],'global_norm')@state['global_down.weight'].T+state['global_down.bias'];a*=ndtr(a)
    b=norm(x[:,256:512],'local_norm')@state['local_down.weight'].T+state['local_down.bias'];b*=ndtr(b)
    pair=a*b if layer.product else (a+b)/math.sqrt(2)
    return .1*norm(pair,'pair_norm')@state['readout.weight'].T


def main():
    torch.set_num_threads(4);torch.manual_seed(331);x=torch.randn(5,640,dtype=torch.float64)
    for rank in [16,32]:
        counts=[]
        for product in [False,True]:
            torch.manual_seed(333);layer=CrossEvidence(rank,product).double()
            counts.append(sum(p.numel() for p in layer.parameters()))
            with torch.no_grad():layer.readout.weight.normal_(std=.04)
            delta=layer(x);expected=independent_cross(layer,x.numpy())
            assert np.allclose(delta.detach().numpy(),expected,atol=1e-12,rtol=1e-12)
            shifted=x.clone();shifted[:,:256]+=7.;shifted[:,256:512]-=3.
            assert torch.allclose(layer(shifted),delta,atol=1e-12)
            auxiliary=x.clone();auxiliary[:,512:]=0.;assert torch.equal(layer(auxiliary),delta)
            for special in [torch.zeros_like(x),torch.full_like(x,1e4)]:
                layer.zero_grad();output=layer(special);assert torch.isfinite(output).all()
                output.square().sum().backward();assert all(torch.isfinite(p.grad).all() for p in layer.parameters())
            layer.zero_grad();delta.square().sum().backward()
            for name in ['global_down.weight','local_down.weight','pair_norm.weight']:
                parameter=dict(layer.named_parameters())[name];gradient=parameter.grad.reshape(-1)
                assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
                index=int(gradient.abs().argmax());original=parameter.detach().reshape(-1)[index].item();losses=[]
                for sign in [-1,1]:
                    with torch.no_grad():parameter.reshape(-1)[index]=original+sign*1e-5
                    losses.append(np.square(independent_cross(layer,x.numpy())).sum())
                with torch.no_grad():parameter.reshape(-1)[index]=original
                assert abs((losses[1]-losses[0])/2e-5-gradient[index].item())<2e-8
        assert counts[0]==counts[1]
    frame,features,digest=load_data('multiscale_vit');reference=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())
    c=torch.load(reference['selected']['checkpoints'][0],map_location='cpu',weights_only=False)
    ids=np.flatnonzero(frame.split.eq('train'));tf=frame.iloc[ids]
    data,_=normalize({k:v[ids[:64]] for k,v in features.items()},None,c['stats']);tx={k:torch.from_numpy(v) for k,v in data.items()}
    y=torch.tensor(tf.label.to_numpy()[:64].copy()).long();disease=torch.tensor([c['diseases'].index(d) for d in tf.disease[:64]])
    weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[:64]),'patient_ids':torch.tensor(__import__('pandas').factorize(tf.patient_group)[0][:64])}
    for rank in [16,32]:
        for product in [False,True]:
            cfg={**c['cfg'],'cross_evidence_rank':rank,'cross_evidence_product':product}
            torch.manual_seed(337);base=Model(c['cfg'],c['dims'],len(c['diseases']));rng=torch.get_rng_state().clone()
            torch.manual_seed(337);model=CrossEvidenceModel(cfg,c['dims'],len(c['diseases']))
            assert torch.equal(rng,torch.get_rng_state())
            for key,value in base.state_dict().items():assert torch.equal(value,model.state_dict()[key])
            for training in [False,True]:
                base.train(training);model.train(training)
                torch.manual_seed(339);a=base(tx);rng=torch.get_rng_state().clone()
                torch.manual_seed(339);b=model(tx)
                assert torch.equal(rng,torch.get_rng_state()) and torch.equal(a['p'],b['p'])
            assert b['cross_evidence_input'].shape==(64,640)
            loss_fn(b,y,disease,weights,cfg).backward();assert model.cross_evidence.readout.weight.grad.abs().sum()>0
            with torch.no_grad():model.cross_evidence.readout.weight-=.1*model.cross_evidence.readout.weight.grad
            model.zero_grad();loss_fn(model(tx),y,disease,weights,cfg).backward()
            for branch in [model.cross_evidence.global_down,model.cross_evidence.local_down]:assert branch.weight.grad.abs().sum()>0
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            restored=CrossEvidenceModel(cfg,c['dims'],len(c['diseases']));restored.load_state_dict(model.state_dict(),strict=True)
            model.eval();restored.eval();assert torch.equal(model(tx)['p'],restored(tx)['p'])
    dump(ROOT/'verification_cross_evidence.json',{'passed':True,'manifest_sha256':digest,'checks':[
        'Double-precision independent NumPy/SciPy complete source norms, GELU, additive/product pair norm and residual match',
        'Selected numerical global/local/norm derivatives, constant/extreme inputs, source shift invariance and unused auxiliary coordinates verified',
        'Additive/product have identical parameter counts within rank16/32; all common M5 initialization, RNG and initial train/eval probabilities exact',
        'Original task gradients reach readout then both source projections; complete strict model restoration exact']})
    print('Cross-source evidence: formulas, gradients, matching controls, identity and restoration passed',flush=True)


if __name__=='__main__':main()
