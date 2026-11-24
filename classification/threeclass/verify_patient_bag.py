"""Independent bag probabilities, weights, sampling, gradients and export checks."""
import json
import numpy as np
import torch
from experiment import ROOT, Model, load_data, normalize, sample_weights, loss_fn, dump
from patient_bag import (make_bags, complete_batches, PatientBagModel,
                         bag_distributions, bag_auxiliary, patient_bag_loss)


def numpy_bags(prob, fine, score, ids, pooling):
    result = []; disease = []; attention = np.zeros(len(ids))
    for token in sorted(set(ids)):
        mask = ids == token
        s = score[mask] if pooling == 'attention' else np.zeros(mask.sum())
        weights = np.exp(s - s.max()); weights /= weights.sum()
        attention[mask] = weights
        result.append((prob[mask] * weights[:, None]).sum(0))
        disease.append((fine[mask] * weights[:, None]).sum(0))
    return np.stack(result), np.stack(disease), attention


def main():
    torch.set_num_threads(4); torch.manual_seed(139)
    ids = torch.tensor([1, 1, 1, 7, 7, 9]); y = torch.tensor([0, 0, 0, 1, 1, 2])
    disease = torch.tensor([0, 0, 0, 3, 3, 4])
    fine = torch.randn(6, 5, dtype=torch.float64).softmax(-1)
    mapping = torch.nn.functional.one_hot(torch.tensor([0, 0, 1, 1, 2]), 3).double()
    logits = fine.log().requires_grad_(); score = torch.randn(6, dtype=torch.float64, requires_grad=True)
    prob = fine @ mapping
    out = {'p': prob, 'subtype': logits, 'bag_quality': score}
    weights = {'bag_ids': ids, 'bag_weight': torch.tensor([.7, .7, .7, 1.4, 1.4, .9], dtype=torch.float64)}
    cfg = {'patient_bag_pool': 'attention', 'label_smoothing': .03, 'aux': .8}
    for pool in ['mean', 'attention']:
        actual = bag_distributions(out, ids, pool)
        expected = numpy_bags(prob.numpy(), fine.numpy(), score.detach().numpy(), ids.numpy(), pool)
        for k, value in zip(['coarse', 'disease', 'attention'], expected):
            assert np.allclose(actual[k].detach().numpy(), value, atol=1e-12, rtol=1e-12)
        assert torch.allclose(actual['coarse'], actual['disease'] @ mapping, atol=1e-12)
        cfg['patient_bag_pool'] = pool
        nll = -.97 * np.log(expected[0][np.arange(3), [0, 1, 2]]) - .03 * np.log(expected[0]).mean(1)
        expected_loss = ((nll - .8 * np.log(expected[1][np.arange(3), [0, 3, 4]])) * np.array([.7, 1.4, .9])).mean()
        assert abs(bag_auxiliary(out, y, disease, weights, cfg).item() - expected_loss) < 1e-12
        order = torch.tensor([5, 3, 2, 0, 4, 1])
        shuffled = {k: v[order] for k, v in out.items()}
        reordered = bag_distributions(shuffled, ids[order], pool)
        assert torch.allclose(reordered['coarse'], actual['coarse'], atol=1e-12)
    cfg['patient_bag_pool'] = 'attention'; loss = bag_auxiliary(out, y, disease, weights, cfg)
    loss.backward(); assert torch.isfinite(score.grad).all() and score.grad[:5].abs().sum() > 0
    assert score.grad[5] == 0 and torch.isfinite(logits.grad).all()
    for index in [0, 2, 4]:
        numeric = []
        for sign in [-1, 1]:
            modified = score.detach().numpy().copy(); modified[index] += sign * 1e-5
            p, f, _ = numpy_bags(prob.numpy(), fine.numpy(), modified, ids.numpy(), 'attention')
            l = -.97 * np.log(p[np.arange(3), [0, 1, 2]]) - .03 * np.log(p).mean(1) - .8 * np.log(f[np.arange(3), [0, 3, 4]])
            numeric.append((l * np.array([.7, 1.4, .9])).mean())
        assert abs((numeric[1] - numeric[0]) / 2e-5 - score.grad[index].item()) < 1e-8
    shifted = {**out, 'bag_quality': score.detach() + torch.tensor([1000., 1000., 1000., -1000., -1000., 42.])}
    assert torch.allclose(bag_distributions(shifted, ids, 'attention')['coarse'], bag_distributions(out, ids, 'attention')['coarse'], atol=1e-12)

    frame, features, digest = load_data('multiscale_vit')
    training_indices = np.flatnonzero(frame.split.eq('train'))
    tf = frame.iloc[training_indices].reset_index(drop=True)
    bag_ids, members, support = make_bags(tf)
    assert len(support) == 388 and support.images.sum() == 1717
    totals = support.groupby(['patient_group', 'label']).weight.sum()
    for label in [0, 1, 2]:
        values = totals[totals.index.get_level_values('label') == label].to_numpy()
        assert np.allclose(values, values[0], atol=1e-6)
    assert np.allclose(support.groupby('label').weight.sum().to_numpy(), len(support) / 3, atol=1e-5)
    for size in [10, 256]:
        for seed in [1, 2]:
            batches = list(complete_batches(members, np.random.default_rng(seed).permutation(len(members)), size))
            assert sorted(np.concatenate(batches)) == list(range(len(tf)))
            seen = set()
            for batch in batches:
                tokens = set(bag_ids[batch]); assert not seen & tokens; seen |= tokens
                assert all((bag_ids[batch] == token).sum() == len(members[token]) for token in tokens)
                assert len(batch) <= size or len(tokens) == 1

    selected = json.loads((ROOT / 'results/milestone5/selection_before_test.json').read_text())['selected']
    c = torch.load(selected['checkpoints'][0], map_location='cpu', weights_only=False)
    chosen = np.concatenate(members[:10])
    data, _ = normalize({k: v[training_indices[chosen]] for k, v in features.items()}, None, c['stats'])
    assert np.array_equal(frame.iloc[training_indices[chosen]].disease, tf.iloc[chosen].disease)
    x = {k: torch.from_numpy(v) for k, v in data.items()}
    yy = torch.tensor(tf.iloc[chosen].label.to_numpy().copy()).long()
    dd = torch.tensor([c['diseases'].index(d) for d in tf.iloc[chosen].disease])
    w = {'leaf': torch.from_numpy(sample_weights(tf, tf.label)[chosen]),
         'patient_ids': torch.tensor(__import__('pandas').factorize(tf.patient_group)[0][chosen]),
         'bag_ids': torch.from_numpy(bag_ids[chosen]), 'bag_weight': torch.from_numpy(support.weight.to_numpy()[bag_ids[chosen]])}
    cfg = {**c['cfg'], 'patient_bag_pool': 'attention', 'patient_bag_weight': .2}
    torch.manual_seed(141); base = Model(c['cfg'], c['dims'], len(c['diseases'])); base_rng = torch.get_rng_state().clone()
    torch.manual_seed(141); bag_model = PatientBagModel(cfg, c['dims'], len(c['diseases']))
    assert torch.equal(base_rng, torch.get_rng_state())
    for key, value in base.state_dict().items():
        assert torch.equal(value, bag_model.state_dict()[key])
    for training in [False, True]:
        base.train(training); bag_model.train(training)
        torch.manual_seed(143); a = base(x); base_rng = torch.get_rng_state().clone()
        torch.manual_seed(143); b = bag_model(x)
        assert torch.equal(base_rng, torch.get_rng_state()) and torch.equal(a['p'], b['p'])
        assert torch.equal(loss_fn(b, yy, dd, w, cfg), patient_bag_loss(b, yy, dd, w, {**cfg, 'patient_bag_weight': 0.}))
    bag_model.zero_grad(); output = bag_model(x); patient_bag_loss(output, yy, dd, w, cfg).backward()
    assert bag_model.bag_quality.weight.grad.abs().sum() > 0 and bag_model.encoder[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in bag_model.parameters())
    exported = {k: v for k, v in bag_model.state_dict().items() if not k.startswith('bag_quality.')}
    base.load_state_dict(exported, strict=True); base.eval(); bag_model.eval()
    assert torch.equal(base(x)['p'], bag_model(x)['p'])
    dump(ROOT / 'verification_patient_bag.json', {'passed': True, 'manifest_sha256': digest,
        'checks': ['Independent NumPy mean/attention bag probabilities, fine marginalization and full weighted auxiliary loss verified',
            'Permutation and within-bag score-shift invariance, singleton behavior and selected numerical score derivatives checked',
            'Patient/class bag weights independently verified; complete group batches exactly cover all images once without splitting bags',
            'Common initialization, train/eval predictions and RNG match M5 exactly; zero auxiliary loss exactly matches original task',
            'Auxiliary supervision reaches quality router and shared encoder; all gradients finite',
            'Dropping training-only quality head strictly restores original M5 single-image inference']})
    print('Patient/disease bags, complete sampling, auxiliary loss, gradients and plain M5 export verified', flush=True)


if __name__ == '__main__':
    main()
