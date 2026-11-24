"""Independent scalar Sinkhorn, full branch reconstruction and task gradients."""
import json
import math
import numpy as np
from scipy.special import ndtr, softmax, logsumexp
import torch
from experiment import ROOT, Model, load_data, normalize, sample_weights, loss_fn, dump
from transport_slots import TransportSlots, TransportModel, slot_weights


def independent_weights(scores, mask=None, balanced=True):
    scores = np.asarray(scores, dtype='float64')
    result = np.zeros_like(scores)
    valid = np.ones(scores.shape[:2], dtype=bool) if mask is None else np.asarray(mask) > .5
    for row in range(len(scores)):
        if not valid[row].any(): valid[row] = True
        ids = np.flatnonzero(valid[row]); s = scores[row, ids]; n, k = s.shape
        if not balanced:
            result[row, ids] = softmax(s, axis=0)
            continue
        v = np.zeros(k)
        for _ in range(30):
            u = np.array([-np.log(n) - logsumexp(s[t] + v) for t in range(n)])
            v = np.array([-np.log(k) - logsumexp(s[:, j] + u) for j in range(k)])
        result[row, ids] = np.exp(s + u[:, None] + v[None]) * k
    return result


def independent_batch_weights(scores, mask=None, balanced=True):
    scores=np.asarray(scores,dtype='float64')
    valid=np.ones(scores.shape[:2],dtype=bool) if mask is None else np.asarray(mask)>.5
    valid=np.where(valid.any(1,keepdims=True),valid,True)
    q=np.where(valid[:,:,None],scores,-np.inf)
    if not balanced:return softmax(q,axis=1)
    for _ in range(30):
        row=logsumexp(q,axis=2,keepdims=True)
        row=np.where(valid[:,:,None],row,0.)
        q=q-row-np.log(valid.sum(1))[:,None,None]
        q=q-logsumexp(q,axis=1,keepdims=True)-np.log(scores.shape[-1])
    return np.exp(q)*scores.shape[-1]


def independent_slots(layer, tokens, mask=None, scalar=False):
    state = {k:v.detach().cpu().numpy().astype('float64') for k,v in layer.state_dict().items()}
    x = np.asarray(tokens, dtype='float64')
    z = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
    z = z * state['norm.weight'] + state['norm.bias']
    key = z @ state['key.weight'].T + state['key.bias']
    scores = 5 * np.tanh((key @ state['queries'].T / math.sqrt(layer.rank)) / 5)
    weights = (independent_weights if scalar else independent_batch_weights)(scores, mask, layer.balanced)
    value = z @ state['value.weight'].T + state['value.bias']; value *= ndtr(value)
    summaries = np.stack([weights[i].T @ value[i] for i in range(len(x))])
    delta = .1 * summaries.reshape(len(x), -1) @ state['readout.weight'].T
    return delta, weights, summaries


def main():
    torch.set_num_threads(4); torch.manual_seed(311)
    x = torch.randn(3, 9, 128, dtype=torch.float64)
    mask = torch.tensor([[1]*9, [1,0,0,0,0,0,0,0,0], [0]*9])
    perm = torch.tensor([7,4,1,8,5,2,6,3,0])
    for slots in [2, 4]:
        variants=[]
        for balanced in [False, True]:
            torch.manual_seed(313); layer = TransportSlots(slots,16,balanced).double()
            with torch.no_grad():
                layer.readout.weight.normal_(std=.04); layer.queries.normal_(std=1.)
            variants.append(sum(p.numel() for p in layer.parameters()))
            delta, weights, summary, scores, valid = layer(x, mask)
            expected = independent_slots(layer, x.numpy(), mask.numpy(),scalar=True)
            batched = independent_slots(layer,x.numpy(),mask.numpy())
            for a,b in zip(expected,batched):assert np.allclose(a,b,atol=2e-12,rtol=2e-12)
            for a,b in zip([delta,weights,summary],expected):
                assert np.allclose(a.detach().numpy(), b, atol=2e-12, rtol=2e-12)
            assert torch.allclose(weights.sum(1),torch.ones((3,slots),dtype=torch.float64),atol=1e-12)
            assert torch.equal(weights[1,1:],torch.zeros_like(weights[1,1:]))
            if balanced:
                target = slots / valid.sum(1).to(torch.float64)
                assert torch.max(torch.abs(weights.sum(2) - target[:,None]) * valid) < 1e-10
            shifted = slot_weights(scores.detach()+4.7,mask,balanced)[0]
            assert torch.allclose(shifted, weights, atol=1e-12)
            shuffled = layer(x[:,perm],mask[:,perm])
            assert torch.allclose(shuffled[0],delta,atol=1e-12)
            assert torch.allclose(shuffled[1],weights[:,perm],atol=1e-12)
            singleton = layer(x[:,:1]); assert torch.allclose(singleton[1],torch.ones_like(singleton[1]))
            # Numerically differentiate the independent formula for actual score/value parameters.
            delta.square().sum().backward()
            parameters=dict(layer.named_parameters())
            for name in ['queries','key.weight','value.weight']:
                parameter=parameters[name];gradient=parameter.grad.reshape(-1)
                assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
                index=int(gradient.abs().argmax());original=parameter.detach().reshape(-1)[index].item();losses=[]
                for sign in [-1,1]:
                    with torch.no_grad():parameter.reshape(-1)[index]=original+sign*1e-5
                    losses.append(np.square(independent_slots(layer,x.numpy(),mask.numpy())[0]).sum())
                with torch.no_grad():parameter.reshape(-1)[index]=original
                assert abs((losses[1]-losses[0])/2e-5-gradient[index].item())<2e-8
            extreme = torch.tensor([[[1e4,-1e4]* (slots//2),[-1e4,1e4]*(slots//2)]],dtype=torch.float64,requires_grad=True)
            extreme_weight=slot_weights(extreme,None,balanced)[0]
            assert torch.isfinite(extreme_weight).all() and torch.allclose(extreme_weight.sum(1),torch.ones((1,slots),dtype=torch.float64))
            extreme_weight.square().sum().backward();assert torch.isfinite(extreme.grad).all()
        assert variants[0]==variants[1]
    frame,features,digest=load_data('multiscale_vit')
    reference=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())
    c=torch.load(reference['selected']['checkpoints'][0],map_location='cpu',weights_only=False)
    ids=np.flatnonzero(frame.split.eq('train'));tf=frame.iloc[ids]
    data,_=normalize({k:v[ids[:64]] for k,v in features.items()},None,c['stats']);tx={k:torch.from_numpy(v) for k,v in data.items()}
    y=torch.tensor(tf.label.to_numpy()[:64].copy()).long();disease=torch.tensor([c['diseases'].index(d) for d in tf.disease[:64]])
    weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[:64]),'patient_ids':torch.tensor(__import__('pandas').factorize(tf.patient_group)[0][:64])}
    for slots in [2,4]:
        for balanced in [False,True]:
            cfg={**c['cfg'],'transport_slots':slots,'transport_balanced':balanced}
            torch.manual_seed(317);base=Model(c['cfg'],c['dims'],len(c['diseases']));rng=torch.get_rng_state().clone()
            torch.manual_seed(317);model=TransportModel(cfg,c['dims'],len(c['diseases']))
            assert torch.equal(rng,torch.get_rng_state())
            for key,value in base.state_dict().items():assert torch.equal(value,model.state_dict()[key])
            for training in [False,True]:
                base.train(training);model.train(training)
                torch.manual_seed(319);a=base(tx);rng=torch.get_rng_state().clone()
                torch.manual_seed(319);b=model(tx)
                assert torch.equal(rng,torch.get_rng_state()) and torch.equal(a['p'],b['p'])
            loss_fn(b,y,disease,weights,cfg).backward();assert model.transport_slots.readout.weight.grad.abs().sum()>0
            with torch.no_grad():model.transport_slots.readout.weight-=.1*model.transport_slots.readout.weight.grad
            model.zero_grad();loss_fn(model(tx),y,disease,weights,cfg).backward()
            for name in ['queries','key.weight','value.weight']:
                assert dict(model.transport_slots.named_parameters())[name].grad.abs().sum()>0
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            restored=TransportModel(cfg,c['dims'],len(c['diseases']));restored.load_state_dict(model.state_dict(),strict=True)
            model.eval();restored.eval();assert torch.equal(model(tx)['p'],restored(tx)['p'])
    dump(ROOT/'verification_transport_slots.json',{'passed':True,'manifest_sha256':digest,'checks':[
        'Independent per-image scalar NumPy/SciPy thirty-iteration Sinkhorn, weights, summaries and residuals match double precision',
        'Matched softmax/Sinkhorn parameter counts, numerical query/key/value derivatives, slot and region masses verified',
        'Permutation invariance, score shifts, partial/all masks, singleton and extreme-score finite forward/backward checked',
        'All four models preserve exact original common initialization, RNG and initial training/evaluation probabilities',
        'Actual original task gradients reach readout and then query/key/value; strict complete model restoration exact']})
    print('Transport slots: independent formulas, numerical gradients, mass constraints, identity and restoration passed',flush=True)


if __name__=='__main__':main()
