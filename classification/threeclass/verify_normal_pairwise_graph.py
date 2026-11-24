"""Full-task gradient, graph gauge, conditional-odds and actual CUDA preflight."""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment import ROOT, KEYS, normalize, loss_fn, sample_weights, dump
from normal_subspace import NormalSuppressedModel, fit_normal_geometry
from normal_pairwise_graph import NormalPairwiseGraphModel, pairwise_graph_loss, pairwise_penalty, INCIDENCE
from audit_normal_pairwise_graph import ordinary, independent_forward, independent_penalty, audit_training_witness
from audit_normal_tangent import complete_task
from verify_normal_subspace import independent_geometry
from kernel_residual_experiments import sha
from normal_subspace_experiments import load_data


def main():
    torch.set_num_threads(4); frame, features, digest = load_data('multiscale_vit')
    for value in features.values(): value[2091:] = np.nan
    fit = np.flatnonzero(frame.split.eq('train')); mask = np.zeros(len(frame), bool); mask[fit] = True
    normalized, stats = normalize(features, mask); data = {key: value[fit] for key, value in normalized.items()}
    tf = frame.iloc[fit].reset_index(drop=True); geometry = fit_normal_geometry(tf, data)
    diagnostics = independent_geometry(tf, data, geometry)
    source = torch.load(ROOT/'runs/normal_subspace/normal_rank32_full/validation/seed_0/model.pt', map_location='cpu', weights_only=False)
    ids = np.random.default_rng(20285101).choice(len(tf), 256, replace=False); raw = {key: value[ids] for key, value in data.items()}
    x = {key: torch.from_numpy(value) for key, value in raw.items()}; mapping = {name: i for i, name in enumerate(source['diseases'])}
    y = torch.tensor(tf.iloc[ids].label.to_numpy()); disease = torch.tensor(tf.iloc[ids].disease.map(mapping).to_numpy())
    weights = {'leaf': torch.from_numpy(sample_weights(tf, tf.label)[ids]), 'patient_ids': torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[ids])}
    cases = []
    for rank in [16, 32]:
        counts = []
        for kind in ['node', 'edge']:
            cfg = {**source['cfg'], 'pairwise_rank': rank, 'pairwise_kind': kind, 'pairwise_aux': .2}
            torch.manual_seed(20285102); base = NormalSuppressedModel(cfg, source['dims'], len(source['diseases']), geometry); expected_rng = torch.random.get_rng_state().clone()
            torch.manual_seed(20285102); model = NormalPairwiseGraphModel(cfg, source['dims'], len(source['diseases']), geometry)
            assert torch.equal(expected_rng, torch.random.get_rng_state())
            for name, value in base.state_dict().items(): assert torch.equal(value, model.state_dict()[name])
            rng = torch.random.get_rng_state().clone(); out = model(x); loss = pairwise_graph_loss(out, y, disease, weights, cfg); loss.backward(); after = torch.random.get_rng_state().clone()
            torch.random.set_rng_state(rng); original_out = base(x); original = loss_fn(original_out, y, disease, weights, cfg); original.backward()
            assert torch.equal(after, torch.random.get_rng_state()) and torch.equal(original_out['p'], out['p'])
            for name, p in base.named_parameters(): assert torch.allclose(p.grad, dict(model.named_parameters())[name].grad, atol=1e-8, rtol=1e-6), name
            for expert in model.pairwise.experts: assert expert[-1].weight.grad.abs().sum() > 0
            state = model.state_dict(); state.update(source['state_dict']); model.load_state_dict(state, strict=True)
            with torch.no_grad():
                for expert in model.pairwise.experts:
                    last = expert[-1]; last.weight.copy_(torch.sin(torch.arange(last.weight.numel()).reshape(last.weight.shape)*.17)*.05)
                    last.bias.copy_(torch.tensor([.03, -.02]))
            model.zero_grad(); state = model.state_dict(keep_vars=True)
            independent = {key: value.detach().clone().requires_grad_(key.startswith('pairwise.')) for key, value in state.items()}
            plain = ordinary({**source, 'cfg': cfg, 'state_dict': state}).train()
            rng = torch.random.get_rng_state().clone(); out = model(x); actual = pairwise_graph_loss(out, y, disease, weights, cfg); actual.backward(); after = torch.random.get_rng_state().clone()
            torch.random.set_rng_state(rng); manual_out = independent_forward(plain, raw, independent, cfg)
            binary = independent_penalty(manual_out['pairwise_binary_logits'], y, weights['leaf'], cfg['label_smoothing'])
            manual = complete_task(manual_out, y, disease, weights, cfg)+cfg['pairwise_aux']*binary; manual.backward()
            assert torch.equal(after, torch.random.get_rng_state()) and torch.allclose(actual, manual, atol=6e-6, rtol=6e-5)
            errors = []
            for name, p in model.named_parameters():
                other = independent[name].grad if name.startswith('pairwise.') else dict(plain.named_parameters())[name].grad
                assert p.grad is not None and other is not None and torch.allclose(p.grad, other, atol=1e-5, rtol=2e-3), name
                errors.append(float((p.grad-other).abs().max()))
            production = pairwise_penalty(out['pairwise_binary_logits'], y, weights['leaf'], .03)
            assert torch.allclose(production, binary, atol=3e-6, rtol=3e-6)
            assert torch.allclose(production, pairwise_penalty(out['pairwise_binary_logits'], y, weights['leaf']*7, .03), atol=3e-6, rtol=3e-6)
            # Original conditional disease probabilities survive every class-constant correction.
            model.eval(); base.load_state_dict({key: value for key, value in model.state_dict().items() if not key.startswith('pairwise.')}, strict=True); base.eval()
            with torch.no_grad():
                changed = model(x); original_out = base(x); logits = changed['subtype']; parent = model.subtype_to_class.argmax(-1)
                odds_errors = []
                for c in range(3):
                    a = logits[:, parent == c].log_softmax(-1); b = original_out['subtype'][:, parent == c].log_softmax(-1)
                    odds_errors.append(float((a-b).abs().max())); assert torch.allclose(a, b, atol=8e-6, rtol=8e-6)
            frozen = {key: value.clone() for key, value in model.named_buffers()}; before = model.pairwise.experts[0][1].weight.detach().clone()
            torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.1).step(); assert not torch.equal(before, model.pairwise.experts[0][1].weight)
            for name, value in frozen.items(): assert torch.equal(value, dict(model.named_buffers())[name])
            restored = NormalPairwiseGraphModel(cfg, source['dims'], len(source['diseases'])).eval(); restored.load_state_dict(model.state_dict(), strict=True)
            with torch.no_grad(): assert torch.equal(model(x)['p'], restored(x)['p'])
            counts.append(sum(p.numel() for p in model.parameters())); cases.append({'rank': rank, 'kind': kind, 'batch_size': 256, 'parameters': counts[-1],
                'maximum_complete_task_gradient_error': max(errors), 'maximum_conditional_disease_logodds_error': max(odds_errors)})
        assert counts[0] == counts[1]
    incidence = torch.tensor(INCIDENCE, dtype=torch.float64); residual = torch.randn(17, 3, dtype=torch.float64)
    shift = residual@incidence.T/3
    least_squares = residual@torch.linalg.pinv(incidence)
    assert torch.allclose(shift, least_squares, atol=1e-12, rtol=1e-12)
    assert torch.allclose(shift, (residual+torch.randn(17, 1)*torch.tensor([1., -1., 1.]))@incidence.T/3, atol=1e-7, rtol=1e-7)
    assert torch.allclose(shift.sum(-1), torch.zeros(17, dtype=torch.float64), atol=1e-12)
    witnesses = []
    for rank in [16, 32]:
        for kind in ['node', 'edge']:
            cfg = {**source['cfg'], 'pairwise_rank': rank, 'pairwise_kind': kind, 'pairwise_aux': .2, 'seed': 0}
            with tempfile.TemporaryDirectory(prefix='normal-pairwise-preflight-') as directory:
                folder = Path(directory); torch.manual_seed(20261124); torch.cuda.manual_seed_all(20261124)
                model = NormalPairwiseGraphModel(cfg, source['dims'], len(source['diseases']), geometry, folder/'training_input_witness.npz').cuda().train()
                local = torch.randperm(len(tf), device='cuda')[:256].cpu().numpy(); gx = {key: torch.from_numpy(value[local]).cuda() for key, value in data.items()}
                gy = torch.tensor(tf.iloc[local].label.to_numpy(), device='cuda'); gd = torch.tensor(tf.iloc[local].disease.map(mapping).to_numpy(), device='cuda')
                gw = {'leaf': torch.from_numpy(sample_weights(tf, tf.label)[local]).cuda(), 'patient_ids': torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[local]).cuda()}
                loss = pairwise_graph_loss(model(gx), gy, gd, gw, cfg); loss.backward()
                c = {'cfg': cfg, 'dims': source['dims'], 'diseases': source['diseases'], 'stats': stats,
                    'state_dict': {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}}
                witnesses.append({'rank': rank, 'kind': kind, 'witness': audit_training_witness(c, folder, features, frame, fit)})
                del model; torch.cuda.empty_cache()
    dump(ROOT/'verification_normal_pairwise_graph.json', {'passed': True, 'manifest_sha256': digest, 'cases': cases,
        'normal_geometry': diagnostics, 'actual_cuda_full_task_witnesses': witnesses,
        'code_sha256': {name: sha(ROOT/name) for name in ['normal_pairwise_graph.py', 'audit_normal_pairwise_graph.py', 'verify_normal_pairwise_graph.py']},
        'checks': ['Four trainedstate256image complete objectives/allparametergradients independently recovered',
            'Zero initialization preserves original probability/basegradients/parameters/RNG; equalparameter node and edge controls',
            'Graph solution agrees with pseudoinverse, cycle gauge vanishes and shifts sum to zero; withinclass conditional disease odds preserved',
            'Patientweight scaling, nonzero optimizer updates, frozenbuffers/strictrestore and four actual CUDA firsttasks/allgradients independently recovered'],
        'external_test_evaluated': False})
    print('Four pairwise graph full-task gradient cases and all four actual CUDA witnesses passed.', flush=True)


if __name__ == '__main__': main()
