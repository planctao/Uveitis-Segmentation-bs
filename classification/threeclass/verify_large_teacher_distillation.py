"""Independent loss formulas, split-local target statistics and export checks."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from experiment import ROOT, Model, load_data, normalize, loss_fn, dump
from large_teacher_distillation import DistilledModel, prepare_teacher, distillation_loss


def numpy_loss(prediction, teacher, patient, weights, mode):
    if mode == 'feature':
        z = prediction.reshape(-1, 2, 1024)
        z = z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-12)
        return ((1 - (z * teacher).sum(-1)).mean(1) * weights).mean()
    z = prediction / np.maximum(np.linalg.norm(prediction, axis=-1, keepdims=True), 1e-12)
    target = sum(teacher[:, v] @ teacher[:, v].T for v in range(2)) / 2
    pair_weights = np.sqrt(weights[:, None] * weights[None, :]) * (patient[:, None] != patient[None, :])
    return (((z @ z.T - target) ** 2 * pair_weights).sum() / max(pair_weights.sum(), 1e-8))


def main():
    torch.set_num_threads(4)
    rng = np.random.default_rng(20261126)
    raw = rng.normal(size=(12, 2048)).astype('float32')
    fit = np.array([0, 2, 4, 5, 8, 9]); targets, stats = prepare_teacher(raw, fit)
    changed = raw.copy(); changed[[1, 3, 6, 7, 10, 11]] = 10000
    other, other_stats = prepare_teacher(changed, fit)
    assert all(np.array_equal(stats[k], other_stats[k]) for k in stats)
    assert np.array_equal(targets[fit], other[fit])
    assert np.allclose(np.linalg.norm(targets, axis=-1), 1, atol=1e-6)
    patients = np.array([0, 0, 1, 2, 2, 3]); weights = np.array([1., .2, 1.3, .5, 2., .6])
    t = torch.tensor(targets[fit], dtype=torch.float64, requires_grad=True)
    # The actual targets are detached inside both losses.
    for mode in ['feature', 'relation']:
        dimensions = 2048 if mode == 'feature' else 7
        z = torch.tensor(rng.normal(size=(6, dimensions)), dtype=torch.float64, requires_grad=True)
        loss = distillation_loss(z, t, torch.tensor(patients), torch.tensor(weights), mode)
        expected = numpy_loss(z.detach().numpy(), t.detach().numpy(), patients, weights, mode)
        assert abs(loss.item() - expected) < 1e-12
        loss.backward(); assert t.grad is None and z.grad.abs().sum() > 0 and torch.isfinite(z.grad).all()
        for row, col in [(0, 1), (2, 3), (3, 5), (5, dimensions - 1)]:
            a = z.detach().numpy().copy(); b = a.copy(); a[row, col] += 1e-5; b[row, col] -= 1e-5
            numeric = (numpy_loss(a, t.detach().numpy(), patients, weights, mode)
                       - numpy_loss(b, t.detach().numpy(), patients, weights, mode)) / 2e-5
            assert abs(numeric - z.grad[row, col].item()) < 1e-8
        perm = np.array([4, 1, 3, 0, 5, 2])
        permuted = distillation_loss(z[perm], t[perm], torch.tensor(patients[perm]), torch.tensor(weights[perm]), mode)
        assert abs(permuted.item() - loss.item()) < 1e-12
    z = torch.randn(6, 7, dtype=torch.float64, requires_grad=True)
    zero = distillation_loss(z, t, torch.zeros(6, dtype=torch.long), torch.ones(6), 'relation')
    assert zero.item() == 0; zero.backward(); assert z.grad.count_nonzero() == 0
    # Duplicating one patient's identical images cannot introduce a self-patient term.
    same = torch.tensor([3, 3]); two = distillation_loss(z[:2], t[:2], same, torch.ones(2), 'relation')
    assert two.item() == 0

    reference = json.loads((ROOT / 'results/milestone5/selection_before_test.json').read_text())['selected']
    checkpoint = torch.load(reference['checkpoints'][0], map_location='cpu', weights_only=False)
    frame, features, digest = load_data('multiscale_vit')
    ids = np.flatnonzero(frame.split.eq('train'))[:16]
    data, _ = normalize({k: v[ids] for k, v in features.items()}, None, checkpoint['stats'])
    inputs = {k: torch.from_numpy(v) for k, v in data.items()}
    y = torch.tensor(frame.iloc[ids].label.to_numpy().copy()).long()
    disease = torch.tensor([checkpoint['diseases'].index(d) for d in frame.iloc[ids].disease])
    w = {'leaf': torch.ones(len(ids)), 'patient_ids': torch.arange(len(ids))}
    for mode in ['feature', 'relation']:
        cfg = {**checkpoint['cfg'], 'large_teacher_mode': mode, 'large_teacher_weight': .2}
        torch.manual_seed(93); base = Model(checkpoint['cfg'], checkpoint['dims'], len(checkpoint['diseases']))
        initial_rng = torch.get_rng_state().clone()
        torch.manual_seed(93); distilled = DistilledModel(cfg, checkpoint['dims'], len(checkpoint['diseases']))
        assert torch.equal(initial_rng, torch.get_rng_state())
        for key, value in base.state_dict().items():
            assert torch.equal(value, distilled.state_dict()[key])
        for training in [False, True]:
            base.train(training); distilled.train(training)
            torch.manual_seed(96); a = base(inputs); base_rng = torch.get_rng_state().clone()
            torch.manual_seed(96); b = distilled(inputs, with_aux=True)
            assert torch.equal(base_rng, torch.get_rng_state())
            assert torch.equal(a['p'], b['p']) and torch.equal(a['fine_embedding'], b['fine_embedding'])
        # Zero distillation preserves one full supervised optimizer update.
        base.train(); distilled.train()
        opt_a = torch.optim.AdamW(base.parameters(), lr=.002, weight_decay=.1)
        opt_b = torch.optim.AdamW(distilled.parameters(), lr=.002, weight_decay=.1)
        opt_a.zero_grad(); torch.manual_seed(98); a = base(inputs)
        loss_fn(a, y, disease, w, checkpoint['cfg']).backward(); torch.nn.utils.clip_grad_norm_(base.parameters(), 5)
        opt_a.step()
        opt_b.zero_grad(); torch.manual_seed(98); b = distilled(inputs, with_aux=True)
        loss_fn(b, y, disease, w, cfg).backward(); torch.nn.utils.clip_grad_norm_(distilled.parameters(), 5)
        opt_b.step()
        for key, value in base.state_dict().items():
            assert torch.equal(value, distilled.state_dict()[key])
        # Auxiliary loss must reach both its decoder and the actual student encoder.
        distilled.zero_grad(); b = distilled(inputs, with_aux=True)
        teacher = F.normalize(torch.randn(len(ids), 2, 1024), dim=-1)
        aux = distillation_loss(b['teacher_prediction'], teacher, w['patient_ids'], w['leaf'], mode)
        aux.backward()
        assert distilled.encoder[0].weight.grad.abs().sum() > 0
        assert distilled.global_proj.weight.grad.abs().sum() > 0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in distilled.semantic_decoder.parameters())
        # Strictly export to original architecture, no teacher or decoder in inference.
        deployed = Model(checkpoint['cfg'], checkpoint['dims'], len(checkpoint['diseases'])).eval()
        deployed.load_state_dict(distilled.inference_state(), strict=True)
        distilled.eval(); assert torch.equal(deployed(inputs)['p'], distilled(inputs)['p'])
        restored = DistilledModel(cfg, checkpoint['dims'], len(checkpoint['diseases'])).eval()
        restored.load_state_dict(distilled.state_dict(), strict=True)
        assert torch.equal(restored(inputs)['p'], distilled(inputs)['p'])
    dump(ROOT / 'verification_large_teacher_distillation.json', {'passed': True, 'manifest_sha256': digest,
        'checks': ['Independent NumPy cosine and interpatient Gram loss formulas match double precision',
            'Selected numerical derivatives and permutation invariance verified for both objectives',
            'Teacher targets detached; all same-patient pairs excluded; one-patient loss differentiably zero',
            'Changing every nonfit teacher row cannot affect fit statistics or targets',
            'Common initialization, train/eval outputs and RNG streams exactly match M5',
            'Zero distillation preserves an entire supervised AdamW step',
            'Nonzero semantic loss reaches decoder, student hidden encoder and global projection',
            'Training checkpoints strictly restore; original M5 inference export gives identical probabilities']})
    print('Large teacher losses, fit-only targets, gradients, RNG and teacher-free export verified', flush=True)


if __name__ == '__main__':
    main()
