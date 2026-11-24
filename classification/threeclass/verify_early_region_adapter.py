"""Independent spatial residual, exact M5 identity, RNG and gradient checks."""
import json
import numpy as np
from scipy.special import ndtr, expit
import torch
from experiment import ROOT, Model, load_data, normalize, loss_fn, dump
from early_region_adapter import EarlyRegionAdapter, EarlyRegionModel


def independent_adapter(layer, early, final):
    state = {k: v.detach().numpy().astype('float64') for k, v in layer.state_dict().items()}
    early = np.asarray(early, dtype='float64'); final = np.asarray(final, dtype='float64')
    def norm(x, prefix):
        return ((x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
                * state[prefix + '.weight'] + state[prefix + '.bias'])
    hidden = norm(early, 'norm') @ state['down.weight'].T + state['down.bias']
    value = hidden * ndtr(hidden) @ state['up.weight'].T + state['up.bias']
    gate = expit(norm(final, 'gate_norm') @ state['gate.weight'].T + state['gate.bias'])
    return .1 * gate * value, gate


def main():
    torch.set_num_threads(4); torch.manual_seed(123)
    early = torch.randn(5, 9, 32, dtype=torch.float64)
    final = torch.randn(5, 9, 128, dtype=torch.float64)
    for rank in [16, 32]:
        layer = EarlyRegionAdapter(32, rank).double()
        assert layer(early, final)[0].count_nonzero() == 0
        optimizer = torch.optim.AdamW(layer.parameters(), lr=.01)
        target = torch.randn_like(final)
        for step in range(3):
            optimizer.zero_grad(); delta, _ = layer(early, final)
            (delta - target).square().mean().backward()
            assert layer.up.weight.grad.abs().sum() > 0
            if step:
                assert layer.down.weight.grad.abs().sum() > 0 and layer.gate.weight.grad.abs().sum() > 0
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in layer.parameters())
            optimizer.step()
        actual = layer(early, final); expected = independent_adapter(layer, early.numpy(), final.numpy())
        for a, b in zip(actual, expected):
            assert np.allclose(a.detach().numpy(), b, atol=1e-12, rtol=1e-12)
        permutation = torch.arange(8, -1, -1)
        assert torch.allclose(layer(early[:, permutation], final[:, permutation])[0], actual[0][:, permutation], atol=1e-12)
        layer.zero_grad(); layer(early, final)[0].square().sum().backward()
        old = layer.gate.weight[0, 3].item(); vals = []
        for sign in [-1, 1]:
            with torch.no_grad():
                layer.gate.weight[0, 3] = old + sign * 1e-5
            vals.append((independent_adapter(layer, early.numpy(), final.numpy())[0] ** 2).sum())
        with torch.no_grad():
            layer.gate.weight[0, 3] = old
        assert abs((vals[1] - vals[0]) / 2e-5 - layer.gate.weight.grad[0, 3].item()) < 1e-8
    ref = json.loads((ROOT / 'results/milestone5/selection_before_test.json').read_text())['selected']
    c = torch.load(ref['checkpoints'][0], map_location='cpu', weights_only=False)
    frame, features, digest = load_data('multiscale_vit'); ids = np.flatnonzero(frame.split.eq('train'))[:16]
    data, _ = normalize({k: v[ids] for k, v in features.items()}, None, c['stats'])
    x = {k: torch.from_numpy(v) for k, v in data.items()}
    x['early_regions'] = torch.randn_like(x['regions'])
    dims = {**c['dims'], 'early_regions': x['early_regions'].shape[-1]}
    y = torch.tensor(frame.iloc[ids].label.to_numpy().copy()).long()
    disease = torch.tensor([c['diseases'].index(d) for d in frame.iloc[ids].disease])
    weights = {'leaf': torch.ones(len(ids)), 'patient_ids': torch.arange(len(ids))}
    for rank in [16, 32]:
        cfg = {**c['cfg'], 'early_region_rank': rank}
        torch.manual_seed(125); base = Model(c['cfg'], c['dims'], len(c['diseases']))
        after_base = torch.get_rng_state().clone()
        torch.manual_seed(125); supplemented = EarlyRegionModel(cfg, dims, len(c['diseases']))
        assert torch.equal(after_base, torch.get_rng_state())
        for key, value in base.state_dict().items():
            assert torch.equal(value, supplemented.state_dict()[key])
        for training in [False, True]:
            base.train(training); supplemented.train(training)
            torch.manual_seed(127); a = base(x); after_base = torch.get_rng_state().clone()
            torch.manual_seed(127); b = supplemented(x)
            assert torch.equal(after_base, torch.get_rng_state())
            assert torch.equal(a['p'], b['p']) and torch.equal(a['fine_embedding'], b['fine_embedding'])
        optimizer = torch.optim.AdamW(supplemented.parameters(), lr=.002)
        for step in range(3):
            optimizer.zero_grad(); loss_fn(supplemented(x), y, disease, weights, cfg).backward()
            assert supplemented.early_adapter.up.weight.grad.abs().sum() > 0
            if step:
                assert supplemented.early_adapter.down.weight.grad.abs().sum() > 0
                assert supplemented.early_adapter.gate.weight.grad.abs().sum() > 0
            optimizer.step()
        supplemented.eval(); restored = EarlyRegionModel(cfg, dims, len(c['diseases'])).eval()
        restored.load_state_dict(supplemented.state_dict(), strict=True)
        assert torch.equal(supplemented(x)['p'], restored(x)['p'])
    dump(ROOT / 'verification_early_region_adapter.json', {'passed': True, 'manifest_sha256': digest,
        'checks': ['Independent double precision LayerNorm/GELU/gated residual formula reconstructed',
            'Joint spatial-region permutation equivariance and numerical gate derivative checked',
            'Zero readouts preserve original M5 train/eval probabilities, contrastive outputs and RNG exactly',
            'Task loss reaches residual readout, intermediate projection and final-conditioned gate',
            'Both full classifiers strictly restore trained checkpoints']})
    print('Intermediate regional residual, identity, gradients and checkpoint restoration verified', flush=True)


if __name__ == '__main__':
    main()
