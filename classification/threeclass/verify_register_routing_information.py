"""Real feature full-task gradients, identical normal-L base and CUDA input replay."""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment import ROOT, KEYS, normalize, loss_fn, sample_weights, dump
from normal_subspace import NormalSuppressedModel, fit_normal_geometry
from large_token_data import load_all
from large_token_evidence import LargeTokenEvidenceModel, fit_side_geometry
from register_routing_information import RegisterRoutingInformationModel, KINDS, routing_regularizer, token_evidence_loss, DETAILS
from audit_register_routing_information import ordinary, independent_forward, independent_side_geometry, audit_training_witness, independent_regularizer, augmented_task
from verify_normal_subspace import independent_geometry
from audit_normal_tangent import complete_task
from kernel_residual_experiments import sha


def verify_penalty():
    generator = torch.Generator().manual_seed(20291401)
    initial = torch.randn(7, 4, dtype=torch.float64, generator=generator)
    patients = torch.tensor([0, 0, 1, 2, 2, 2, 3]); results = []
    for kind in KINDS:
        logits = initial.clone().requires_grad_(); attention = logits.softmax(1)
        a = routing_regularizer(attention, patients, kind); gradient = torch.autograd.grad(a, logits, retain_graph=True)[0]
        b = independent_regularizer(attention, patients, kind); other = torch.autograd.grad(b, logits, retain_graph=True)[0]
        assert torch.allclose(a,b,atol=2e-15,rtol=2e-15) and torch.allclose(gradient,other,atol=2e-15,rtol=2e-15)
        row_order = torch.tensor([6, 3, 0, 2, 5, 1, 4]); duplicated = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0, 1])
        alternatives = [routing_regularizer(attention[row_order], patients[row_order], kind),
            routing_regularizer(attention[:,torch.tensor([3, 1, 0, 2])], patients, kind),
            routing_regularizer(attention, 100-patients*9, kind),
            routing_regularizer(attention[duplicated], patients[duplicated], kind)]
        for alternative in alternatives:
            derivative = torch.autograd.grad(alternative, logits, retain_graph=True)[0]
            assert torch.allclose(a,alternative,atol=2e-15,rtol=2e-15) and torch.allclose(gradient,derivative,atol=2e-15,rtol=2e-15)
        extreme = torch.tensor([[1000., 0., -1000., -500.], [-1000., 0., 1000., -500.]], requires_grad=True)
        value = routing_regularizer(extreme.softmax(1), torch.tensor([0,1]), kind)
        derivative = torch.autograd.grad(value, extreme)[0]
        assert torch.isfinite(value) and torch.isfinite(derivative).all()
        uniform = routing_regularizer(torch.ones(4,4,dtype=torch.float64)/4, torch.arange(4), kind)
        balanced = routing_regularizer(torch.eye(4,dtype=torch.float64), torch.arange(4), kind)
        collapsed = routing_regularizer(torch.tensor([[1.,0,0,0]],dtype=torch.float64).repeat(4,1), torch.arange(4), kind)
        expected = {'control':(0,0,0), 'smooth':(0,1,1), 'balance':(0,0,1), 'information':(1,0,1)}[kind]
        assert torch.allclose(torch.stack([uniform,balanced,collapsed]),torch.tensor(expected,dtype=torch.float64),atol=1e-14)
        results.append({'kind':kind, 'independent_value_error':float((a-b).abs()),'independent_gradient_error':float((gradient-other).abs().max()),
            'patient_id_row_token_and_patientduplication_invariance':True, 'extreme_entropy_gradients_finite':True,'uniform_balanced_collapsed_limits_verified':True})
    return results


def main():
    torch.set_num_threads(4); penalty_checks = verify_penalty(); frame, base_features, token_bank, digest = load_all(); fit = np.flatnonzero(frame.split.eq('train'))
    mask = np.zeros(len(frame), bool); mask[fit] = True; tf = frame.iloc[fit].reset_index(drop=True)
    normalized, base_stats = normalize(base_features, mask); base_geometry = fit_normal_geometry(tf, {key: value[fit] for key, value in normalized.items()})
    base_diagnostic = independent_geometry(tf, {key: value[fit] for key, value in normalized.items()}, base_geometry)
    source = torch.load(ROOT/'runs/normal_large_reference/large_dual/validation/seed_0/model.pt', map_location='cpu', weights_only=False)
    dims = {**source['dims'], 'side_tokens': 2048}; ids = np.random.default_rng(20285201).choice(len(tf), 256, replace=False)
    mapping = {name: i for i, name in enumerate(source['diseases'])}; y = torch.tensor(tf.iloc[ids].label.to_numpy()); disease = torch.tensor(tf.iloc[ids].disease.map(mapping).to_numpy())
    weights = {'leaf': torch.from_numpy(sample_weights(tf, tf.label)[ids]), 'patient_ids': torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[ids])}
    cases = []; witnesses = []; diagnostics = {}; counts = []
    for kind in KINDS:
        stream = 'registers'
        features = {**base_features, 'side_tokens': token_bank[stream]}; normalized, stats = normalize(features, mask)
        geometry = {**base_geometry, **fit_side_geometry(tf, normalized['side_tokens'][fit])}
        diagnostics[stream] = independent_side_geometry(tf, normalized['side_tokens'][fit], geometry)
        raw = {key: value[fit][ids] for key, value in normalized.items()}; x = {key: torch.from_numpy(value) for key, value in raw.items()}
        cfg = {**source['cfg'], 'token_evidence_stream': stream, 'register_routing_information': kind, 'routing_information_weight': 0. if kind == 'control' else .02}
        torch.manual_seed(20285202); base = NormalSuppressedModel(cfg, source['dims'], len(source['diseases']), base_geometry); expected_rng = torch.random.get_rng_state().clone()
        torch.manual_seed(20285202); model = RegisterRoutingInformationModel(cfg, dims, len(source['diseases']), geometry)
        assert torch.equal(expected_rng, torch.random.get_rng_state())
        for name, value in base.state_dict().items(): assert torch.equal(value, model.state_dict()[name])
        rng = torch.random.get_rng_state().clone(); out = model(x); actual = loss_fn(out, y, disease, weights, cfg); actual.backward(); after = torch.random.get_rng_state().clone()
        torch.random.set_rng_state(rng); original = loss_fn(base({key: x[key] for key in KEYS}), y, disease, weights, cfg); original.backward()
        assert torch.equal(actual, original) and torch.equal(after, torch.random.get_rng_state()) and not torch.count_nonzero(out['token_delta'])
        for name, p in base.named_parameters(): assert torch.equal(p.grad, dict(model.named_parameters())[name].grad), name
        assert model.token_evidence.readout.weight.grad.abs().sum() > 0
        zero_readout_routing_gradient = None
        if kind != 'control':
            model.zero_grad(); z = model(x)
            penalty = routing_regularizer(z['token_attention'], weights['patient_ids'], kind)
            gradient = torch.autograd.grad(penalty, (model.token_evidence.query.weight, model.token_evidence.key.weight, model.encoder[0].weight))
            zero_readout_routing_gradient = [float(value.abs().sum()) for value in gradient]
            assert all(value > 1e-8 for value in zero_readout_routing_gradient)
        state = model.state_dict(); state.update(source['state_dict']); model.load_state_dict(state, strict=True)
        control_error = None
        if kind == 'control':
            trained = torch.load(ROOT/'runs/large_token_evidence/registers/validation/seed_0/model.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(trained['state_dict'], strict=True); model.eval()
            original_model = LargeTokenEvidenceModel(trained['cfg'], trained['dims'], len(trained['diseases'])).eval()
            original_model.load_state_dict(trained['state_dict'], strict=True)
            with torch.no_grad():
                control_error = float((model(x)['p']-original_model(x)['p']).abs().max()); assert control_error == 0.
            model.train()
        with torch.no_grad():
            last = model.token_evidence.readout; last.weight.copy_(torch.sin(torch.arange(last.weight.numel()).reshape(last.weight.shape)*.17)*.01)
            last.bias.copy_(torch.linspace(-.01, .01, 256))
        model.zero_grad(); state = model.state_dict(keep_vars=True); independent = {key: value.detach().clone().requires_grad_(key.startswith('token_evidence.')) for key, value in state.items()}
        plain = ordinary({**source, 'cfg': cfg, 'dims': dims, 'state_dict': state}).train()
        rng = torch.random.get_rng_state().clone(); out = model(x); actual = token_evidence_loss(out, y, disease, weights, cfg); actual.backward(); after = torch.random.get_rng_state().clone()
        torch.random.set_rng_state(rng); manual_out = independent_forward(plain, raw, independent); manual = augmented_task(manual_out, y, disease, weights, cfg); manual.backward()
        assert torch.equal(after, torch.random.get_rng_state()) and torch.allclose(actual, manual, atol=6e-6, rtol=6e-5)
        errors = []
        for name, p in model.named_parameters():
            other = independent[name].grad if name.startswith('token_evidence.') else dict(plain.named_parameters())[name].grad
            assert p.grad is not None and other is not None and torch.allclose(p.grad, other, atol=1e-5, rtol=2e-3), name
            errors.append(float((p.grad-other).abs().max()))
        for key in ['p']+DETAILS: assert torch.allclose(out[key], manual_out[key], atol=7e-6, rtol=1e-4), key
        model.eval()
        with torch.no_grad():
            expected = model(x)['p']; permutation = torch.arange(x['side_tokens'].shape[1]-1, -1, -1)
            changed = model({**x, 'side_tokens': x['side_tokens'][:, permutation]})['p']; permutation_error = float((changed-expected).abs().max()); assert permutation_error < 4e-6
            tokens = model._tokens.clone(); hidden = torch.randn(256, 256)
            a = model.token_evidence(tokens, hidden); b = model.token_evidence(tokens, hidden.roll(1, 0))
            query_change = float((a['token_attention']-b['token_attention']).abs().max()); assert query_change > 1e-6
        frozen = {key: value.clone() for key, value in model.named_buffers()}; before = model.token_evidence.key.weight.detach().clone()
        torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.1).step(); assert not torch.equal(before, model.token_evidence.key.weight)
        for name, value in frozen.items(): assert torch.equal(value, dict(model.named_buffers())[name])
        restored = RegisterRoutingInformationModel(cfg, dims, len(source['diseases'])).eval(); restored.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad(): assert torch.equal(model(x)['p'], restored(x)['p'])
        counts.append(sum(p.numel() for p in model.parameters())); cases.append({'kind': kind, 'm8_exact_control_error': control_error, 'zero_readout_routing_query_key_encoder_gradient_l1': zero_readout_routing_gradient, 'stream': stream, 'batch_size': 256, 'tokens': x['side_tokens'].shape[1],
            'parameters': counts[-1], 'maximum_complete_task_gradient_error': max(errors), 'token_permutation_error': permutation_error, 'hidden_query_attention_change': query_change})
        with tempfile.TemporaryDirectory(prefix='large-token-evidence-preflight-') as directory:
            folder = Path(directory); cfg = {**cfg, 'seed': 0}; torch.manual_seed(20261124); torch.cuda.manual_seed_all(20261124)
            model = RegisterRoutingInformationModel(cfg, dims, len(source['diseases']), geometry, folder/'training_input_witness.npz').cuda().train()
            local = torch.randperm(len(tf), device='cuda')[:256].cpu().numpy(); gx = {key: torch.from_numpy(value[fit][local]).cuda() for key, value in normalized.items()}
            gy = torch.tensor(tf.iloc[local].label.to_numpy(), device='cuda'); gd = torch.tensor(tf.iloc[local].disease.map(mapping).to_numpy(), device='cuda')
            gw = {'leaf': torch.from_numpy(sample_weights(tf, tf.label)[local]).cuda(), 'patient_ids': torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[local]).cuda()}
            loss = token_evidence_loss(model(gx), gy, gd, gw, cfg); loss.backward()
            c = {'cfg': cfg, 'dims': dims, 'diseases': source['diseases'], 'stats': stats, 'state_dict': {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}}
            witnesses.append({'kind': kind, 'stream': stream, 'witness': audit_training_witness(c, folder, features, frame, fit)})
            del model; torch.cuda.empty_cache()
    assert len(set(counts)) == 1
    dump(ROOT/'verification_register_routing_information.json', {'passed': True, 'manifest_sha256': digest, 'cases': cases, 'base_normal_geometry': base_diagnostic,
        'routing_information_checks': penalty_checks, 'side_normal_geometries': diagnostics, 'actual_cuda_full_task_witnesses': witnesses,
        'code_sha256': {name: sha(ROOT/name) for name in ['register_routing_information.py', 'audit_register_routing_information.py', 'verify_register_routing_information.py', 'large_token_data.py']},
        'checks': ['Four actual256image augmentedtasks/allparametergradients independently reconstructed',
            'OriginalnormalL initialization/RNG/unregularizedtask/basegradients identical atzeroreadout; modifiedarms explicitly have nonzero routinggradients atzeroreadout; equalparameters and exactM8control',
            'Fitnormalpatient tokenPCA recovered by Gram spectrum; independent projections/query/keys/values/attention/delta',
            'Patient-loop entropy/marginal and allgradients independently recovered; patientIDrename/roworder/tokenpermutation/duplicatedpatient invariance; finite saturated gradients; tokenpermutation/nonzeroquery/optimizerupdates/frozenbuffers/strictrestore and fouractualCUDA augmentedfirsttask/allgradientreplays'],
        'classification_fits': 0, 'classification_scores': 0, 'external_test_evaluated': False})
    print('Four register routing-information augmented-task gradient cases and four actual CUDA witnesses passed.', flush=True)


if __name__ == '__main__': main()
