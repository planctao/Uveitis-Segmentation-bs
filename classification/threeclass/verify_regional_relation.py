"""Independent relation formulas, edge cases and exact initial M5 identity."""
import json
import math
import numpy as np
from scipy.special import ndtr, softmax
import torch
from experiment import ROOT, Model, load_data, normalize, sample_weights, loss_fn, dump
from regional_relation import RegionalRelation, RelationModel


def independent_relation(layer, tokens, mask=None):
    state = {k: v.detach().cpu().numpy().astype('float64') for k, v in layer.state_dict().items()}
    x = np.asarray(tokens, dtype='float64'); n, count, _ = x.shape
    z = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
    z = z * state['norm.weight'] + state['norm.bias']
    value = z @ state['value.weight'].T + state['value.bias']; value = value * ndtr(value)
    score = np.zeros((n, count, count))
    if layer.kind == 'content':
        q = z @ state['query.weight'].T; k = z @ state['key.weight'].T
        score = q @ k.transpose(0, 2, 1) / math.sqrt(layer.rank)
    if 'relative_bias' in state:
        score = score + state['relative_bias'][state['relative_index'].astype('int64')]
    valid = np.ones((n, count), dtype=bool) if mask is None else np.asarray(mask) > .5
    for row in range(n):
        if not valid[row].any(): valid[row] = True
    attention = np.zeros_like(score)
    for row in range(n):
        for query in range(count):
            allowed = valid[row].copy(); allowed[query] = False
            if not allowed.any(): allowed[query] = True
            s = np.where(allowed, score[row, query], -1e4)
            attention[row, query] = softmax(s)
    delta = .1 * ((attention @ value - value) @ state['readout.weight'].T) * valid[:, :, None]
    return delta, attention


def main():
    torch.set_num_threads(4); torch.manual_seed(199)
    x = torch.randn(3, 9, 128, dtype=torch.float64)
    mask = torch.tensor([[1]*9, [1,0,0,0,0,0,0,0,0], [0]*9])
    permutation = torch.tensor([7,4,1,8,5,2,6,3,0])
    for kind, position in [('uniform', False), ('content', False), ('content', True)]:
        layer = RegionalRelation(16, kind, position).double()
        with torch.no_grad():
            layer.readout.weight.normal_(std=.04)
            if position: layer.relative_bias.normal_(std=.1)
        delta, attention = layer(x, mask)
        expected = independent_relation(layer, x.numpy(), mask.numpy())
        assert np.allclose(delta.detach().numpy(), expected[0], atol=1e-12, rtol=1e-12)
        assert np.allclose(attention.detach().numpy(), expected[1], atol=1e-12, rtol=1e-12)
        assert torch.allclose(attention.sum(-1), torch.ones((3,9),dtype=torch.float64))
        assert torch.equal(delta[1], torch.zeros_like(delta[1]))
        constant = x[:, :1].expand(-1,9,-1)
        assert layer(constant)[0].abs().max() < 1e-15
        assert torch.equal(layer(x[:, :1])[0], torch.zeros_like(x[:, :1])) if not position else True
        if not position:
            permuted = layer(x[:,permutation],mask[:,permutation])
            assert torch.allclose(permuted[0], delta[:,permutation], atol=1e-12)
            assert torch.allclose(permuted[1], attention[:,permutation][:,:,permutation], atol=1e-12)
        # Score and relative-position derivatives after a nonzero readout.
        loss = delta.square().sum(); loss.backward()
        names = ['value.weight'] + ([] if kind == 'uniform' else ['query.weight', 'key.weight'])
        if position: names.append('relative_bias')
        parameters = dict(layer.named_parameters())
        for name in names:
            parameter = parameters[name]; gradient = parameter.grad.reshape(-1)
            assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
            index = int(gradient.abs().argmax()); original = parameter.detach().reshape(-1)[index].item(); losses = []
            for sign in [-1, 1]:
                with torch.no_grad(): parameter.reshape(-1)[index] = original + sign * 1e-5
                losses.append(np.square(independent_relation(layer,x.numpy(),mask.numpy())[0]).sum())
            with torch.no_grad(): parameter.reshape(-1)[index] = original
            assert abs((losses[1]-losses[0])/2e-5-gradient[index].item()) < 1e-8

    frame, features, digest = load_data('multiscale_vit')
    reference = json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())
    c = torch.load(reference['selected']['checkpoints'][0],map_location='cpu',weights_only=False)
    training_indices=np.flatnonzero(frame.split.eq('train'))
    tf=frame.iloc[training_indices]
    data,_ = normalize({k:v[training_indices[:64]] for k,v in features.items()},None,c['stats']); tx={k:torch.from_numpy(v) for k,v in data.items()}
    y=torch.tensor(tf.label.to_numpy()[:64].copy()).long()
    disease=torch.tensor([c['diseases'].index(d) for d in tf.disease[:64]])
    weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[:64]),
             'patient_ids':torch.tensor(__import__('pandas').factorize(tf.patient_group)[0][:64])}
    for kind,rank,position in [('uniform',16,False),('content',16,False),('content',32,False),('content',32,True)]:
        cfg={**c['cfg'],'relation_kind':kind,'relation_rank':rank,'relation_position':position}
        torch.manual_seed(201); base=Model(c['cfg'],c['dims'],len(c['diseases'])); rng=torch.get_rng_state().clone()
        torch.manual_seed(201); model=RelationModel(cfg,c['dims'],len(c['diseases']))
        assert torch.equal(rng,torch.get_rng_state())
        for key,value in base.state_dict().items(): assert torch.equal(value,model.state_dict()[key])
        for training in [False,True]:
            base.train(training); model.train(training)
            torch.manual_seed(203);a=base(tx);rng=torch.get_rng_state().clone()
            torch.manual_seed(203);b=model(tx)
            assert torch.equal(rng,torch.get_rng_state()) and torch.equal(a['p'],b['p'])
        loss_fn(b,y,disease,weights,cfg).backward()
        assert model.relation.readout.weight.grad.abs().sum()>0
        with torch.no_grad(): model.relation.readout.weight -= .1*model.relation.readout.weight.grad
        model.zero_grad();loss_fn(model(tx),y,disease,weights,cfg).backward()
        assert model.relation.value.weight.grad.abs().sum()>0
        if kind=='content':assert model.relation.query.weight.grad.abs().sum()>0
        if position:assert model.relation.relative_bias.grad.abs().sum()>0
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        restored=RelationModel(cfg,c['dims'],len(c['diseases']));restored.load_state_dict(model.state_dict(),strict=True)
        model.eval();restored.eval();assert torch.equal(model(tx)['p'],restored(tx)['p'])
    dump(ROOT/'verification_regional_relation.json',{'passed':True,'manifest_sha256':digest,'checks':[
        'Independent double-precision NumPy/SciPy complete relation formula and selected numerical derivatives verified',
        'Uniform/content routing, non-position permutation equivariance, all-masked fallback, singleton and constant-token boundaries verified',
        'All four arms preserve common M5 initialization, initial train/eval predictions and random-number stream exactly',
        'Actual task gradients reach neutral readout then value/query/relative-position parameters; full restoration exact']})
    print('Regional relations: formulas, derivatives, boundaries, initial identity, gradients and strict recovery verified',flush=True)


if __name__=='__main__':main()
