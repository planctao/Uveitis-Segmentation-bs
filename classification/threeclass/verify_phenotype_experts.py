"""Mixture formula, permutation, identity, gradients and classifier integration."""
import json
import numpy as np
from scipy.special import ndtr, softmax
import torch
from experiment import ROOT, Model, load_data, normalize, loss_fn, dump
from phenotype_experts import PhenotypeExperts, PhenotypeModel


def numpy_forward(layer, z, context):
    state = {k: v.detach().numpy() for k, v in layer.state_dict().items()}
    def norm(x, prefix):
        a = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
        return a * state[prefix + '.weight'] + state[prefix + '.bias']
    gates = softmax(norm(context, 'route_norm') @ state['gate.weight'].T + state['gate.bias'], axis=1)
    x = norm(z, 'norm'); values = []
    for i in range(len(layer.down)):
        hidden = x @ state[f'down.{i}.weight'].T + state[f'down.{i}.bias']
        values.append((hidden * ndtr(hidden)) @ state[f'up.{i}.weight'].T + state[f'up.{i}.bias'])
    delta = layer.scale * (gates[:, :, None] * np.stack(values, 1)).sum(1)
    return z + delta, gates, delta


def main():
    torch.set_num_threads(4); torch.manual_seed(83)
    z = torch.randn(8, 16, dtype=torch.float64)
    context = torch.randn_like(z)
    for count in [2, 4]:
        module = PhenotypeExperts(16, 5, count).double()
        assert torch.equal(module(z, context)[0], z)
        optimizer = torch.optim.AdamW(module.parameters(), lr=.01)
        target = torch.randn_like(z)
        for step in range(3):
            optimizer.zero_grad(); output, gate, delta = module(z, context)
            (output - target).square().mean().backward()
            assert all(layer.weight.grad.abs().sum() > 0 for layer in module.up)
            if step:
                assert module.gate.weight.grad.abs().sum() > 0
                assert all(layer.weight.grad.abs().sum() > 0 for layer in module.down)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters())
            optimizer.step()
        actual = module(z, context); expected = numpy_forward(module, z.numpy(), context.numpy())
        for a, b in zip(actual, expected):
            assert np.allclose(a.detach().numpy(), b, atol=1e-12, rtol=1e-12)
        assert torch.allclose(actual[1].sum(1), torch.ones(8, dtype=torch.float64))
        # Renaming experts and the corresponding gate channels preserves output.
        perm = torch.arange(count - 1, -1, -1); state = module.state_dict(); shuffled = dict(state)
        shuffled['gate.weight'] = state['gate.weight'][perm]
        shuffled['gate.bias'] = state['gate.bias'][perm]
        for i, old in enumerate(perm.tolist()):
            for prefix in ['up', 'down']:
                for suffix in ['weight', 'bias']:
                    shuffled[f'{prefix}.{i}.{suffix}'] = state[f'{prefix}.{old}.{suffix}']
        renamed = PhenotypeExperts(16, 5, count).double(); renamed.load_state_dict(shuffled, strict=True)
        assert torch.allclose(actual[0], renamed(z, context)[0], atol=1e-12, rtol=1e-12)
        # Check a routing derivative using the independent NumPy expression.
        module.zero_grad(); module(z, context)[0].square().sum().backward()
        numerical = []; original = module.gate.weight.detach()[0, 1].item()
        for direction in [-1, 1]:
            with torch.no_grad():
                module.gate.weight[0, 1] = original + direction * 1e-5
            numerical.append((numpy_forward(module, z.numpy(), context.numpy())[0] ** 2).sum())
        with torch.no_grad():
            module.gate.weight[0, 1] = original
        assert abs((numerical[1] - numerical[0]) / 2e-5 - module.gate.weight.grad[0, 1].item()) < 1e-8

    selected = json.loads((ROOT / 'results/milestone5/selection_before_test.json').read_text())['selected']
    c = torch.load(selected['checkpoints'][0], map_location='cpu', weights_only=False)
    frame, features, digest = load_data('multiscale_vit')
    ids = np.flatnonzero(frame.split.eq('train'))[:16]
    data, _ = normalize({k: v[ids] for k, v in features.items()}, None, c['stats'])
    x = {k: torch.from_numpy(v) for k, v in data.items()}
    y = torch.tensor(frame.iloc[ids].label.to_numpy().copy()).long()
    disease = torch.tensor([c['diseases'].index(d) for d in frame.iloc[ids].disease])
    weights = {'leaf': torch.ones(len(ids)), 'patient_ids': torch.arange(len(ids))}
    for count in [2, 4]:
        for rank in [16, 32]:
            cfg = {**c['cfg'], 'phenotype_experts': count, 'phenotype_rank': rank}
            torch.manual_seed(93); base = Model(c['cfg'], c['dims'], len(c['diseases']))
            base_rng = torch.get_rng_state().clone()
            torch.manual_seed(93); extended = PhenotypeModel(cfg, c['dims'], len(c['diseases']))
            assert torch.equal(base_rng, torch.get_rng_state())
            for key, value in base.state_dict().items():
                assert torch.equal(value, extended.state_dict()[key])
            for training in [False, True]:
                base.train(training); extended.train(training)
                torch.manual_seed(97); a = base(x); base_rng = torch.get_rng_state().clone()
                torch.manual_seed(97); b = extended(x)
                assert torch.equal(base_rng, torch.get_rng_state())
                assert torch.equal(a['p'], b['p']) and torch.equal(a['fine_embedding'], b['fine_embedding'])
                assert not b['phenotype_delta'].count_nonzero()
            # Router is exactly invariant to two different hidden dropout draws.
            torch.manual_seed(98); first = extended(x)['phenotype_gates']
            torch.manual_seed(99); second = extended(x)['phenotype_gates']
            assert torch.equal(first, second)
            optimizer = torch.optim.AdamW(extended.parameters(), lr=.002)
            for update in range(3):
                optimizer.zero_grad(); loss_fn(extended(x), y, disease, weights, cfg).backward()
                assert all(layer.weight.grad.abs().sum() > 0 for layer in extended.phenotype.up)
                if update:
                    assert extended.phenotype.gate.weight.grad.abs().sum() > 0
                optimizer.step()
            extended.eval(); restored = PhenotypeModel(cfg, c['dims'], len(c['diseases'])).eval()
            restored.load_state_dict(extended.state_dict(), strict=True)
            assert torch.equal(restored(x)['p'], extended(x)['p'])
    dump(ROOT / 'verification_phenotype_experts.json', {'passed': True, 'manifest_sha256': digest,
        'checks': ['Independent double precision NumPy LayerNorm/GELU/softmax mixture reconstructed',
            'Expert/channel permutation invariance and numerical routing derivative verified',
            'Zero readouts preserve exact initial embeddings, probabilities, contrastive embeddings and RNG',
            'Pre-dropout router exactly invariant to hidden dropout draws',
            'All actual expert readouts, bottlenecks and gate receive finite task gradients',
            'All four full classifiers strictly restore trained checkpoints']})
    print('Phenotype mixture formula, identity, invariant routing and full classifier checks passed', flush=True)


if __name__ == '__main__':
    main()
