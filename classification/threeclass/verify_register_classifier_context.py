"""Real feature full-task gradients, identical normal-L base and CUDA input replay."""
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment import ROOT, KEYS, normalize, loss_fn, sample_weights, dump
from normal_subspace import NormalSuppressedModel, fit_normal_geometry
from large_token_data import load_all
from register_classifier_context import KINDS
from large_token_evidence import LargeTokenEvidenceModel
from register_classifier_context import RegisterClassifierContextModel, token_evidence_loss, DETAILS
from large_token_evidence import fit_side_geometry
from audit_register_classifier_context import ordinary, independent_forward, independent_side_geometry, audit_training_witness
from verify_normal_subspace import independent_geometry
from audit_normal_tangent import complete_task
from kernel_residual_experiments import sha


def main():
    torch.set_num_threads(4); frame, base_features, token_bank, digest = load_all(); fit = np.flatnonzero(frame.split.eq('train'))
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
        cfg = {**source['cfg'], 'token_evidence_stream': stream, 'register_classifier_context': kind}
        torch.manual_seed(20285202); base = NormalSuppressedModel(cfg, source['dims'], len(source['diseases']), base_geometry); expected_rng = torch.random.get_rng_state().clone()
        torch.manual_seed(20285202); model = RegisterClassifierContextModel(cfg, dims, len(source['diseases']), geometry)
        assert torch.equal(expected_rng, torch.random.get_rng_state())
        for name, value in base.state_dict().items(): assert torch.equal(value, model.state_dict()[name])
        rng = torch.random.get_rng_state().clone(); out = model(x); actual = loss_fn(out, y, disease, weights, cfg); actual.backward(); after = torch.random.get_rng_state().clone()
        torch.random.set_rng_state(rng); original = loss_fn(base({key: x[key] for key in KEYS}), y, disease, weights, cfg); original.backward()
        assert torch.equal(actual, original) and torch.equal(after, torch.random.get_rng_state()) and not torch.count_nonzero(out['token_delta'])
        for name, p in base.named_parameters(): assert torch.equal(p.grad, dict(model.named_parameters())[name].grad), name
        assert model.token_evidence.readout.weight.grad.abs().sum() > 0
        state = model.state_dict(); state.update(source['state_dict']); model.load_state_dict(state, strict=True)
        control_error = None
        if kind == 'hidden_control':
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
        torch.random.set_rng_state(rng); manual_out = independent_forward(plain, raw, independent, cfg); manual = complete_task(manual_out, y, disease, weights, cfg); manual.backward()
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
            a = model.token_evidence(tokens, hidden, model.flat.weight, model.subtype_to_class); b = model.token_evidence(tokens, hidden.roll(1, 0), model.flat.weight, model.subtype_to_class)
            query_change = float((a['token_attention']-b['token_attention']).abs().max()); assert query_change > 1e-6
            classifier_permutation_error = None
            if kind != 'hidden_control':
                permutation = torch.arange(len(model.flat.weight)-1,-1,-1)
                changed = model.token_evidence(tokens, hidden, model.flat.weight[permutation], model.subtype_to_class[permutation])
                classifier_permutation_error = float((a['token_delta']-changed['token_delta']).abs().max())
                assert classifier_permutation_error < 4e-6
                coarse_counts = model.subtype_to_class.sum(0)
                weights_per_query = (model.subtype_to_class/coarse_counts).sum(1)/3 if kind == 'disease_balanced' else torch.ones(3)/3
                per_query = (a['token_attention'][...,None]*a['token_values'][:,None]).sum(2)
                assert torch.allclose(a['token_pooled'], (per_query*weights_per_query[None,:,None]).sum(1), atol=2e-6,rtol=1e-5)
                if kind == 'disease_balanced':
                    assert torch.allclose(weights_per_query@model.subtype_to_class, torch.ones(3)/3)
        context_gradient_norm = None
        if kind != 'hidden_control':
            with torch.enable_grad():
                context_output = model.token_evidence(tokens, hidden, model.flat.weight, model.subtype_to_class)
                context_gradient = torch.autograd.grad(context_output['token_delta'].square().mean(), model.flat.weight, allow_unused=True)[0]
                if kind == 'class_detached': assert context_gradient is None
                else:
                    context_gradient_norm = float(context_gradient.abs().sum()); assert context_gradient_norm > 1e-8
        frozen = {key: value.clone() for key, value in model.named_buffers()}; before = model.token_evidence.key.weight.detach().clone()
        torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.1).step(); assert not torch.equal(before, model.token_evidence.key.weight)
        for name, value in frozen.items(): assert torch.equal(value, dict(model.named_buffers())[name])
        restored = RegisterClassifierContextModel(cfg, dims, len(source['diseases'])).eval(); restored.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad(): assert torch.equal(model(x)['p'], restored(x)['p'])
        counts.append(sum(p.numel() for p in model.parameters())); cases.append({'kind': kind, 'm8_exact_control_error': control_error, 'classifier_row_permutation_error': classifier_permutation_error, 'context_classifier_gradient_l1': context_gradient_norm, 'batch_size': 256, 'tokens': x['side_tokens'].shape[1],
            'parameters': counts[-1], 'maximum_complete_task_gradient_error': max(errors), 'token_permutation_error': permutation_error, 'hidden_query_attention_change': query_change})
        with tempfile.TemporaryDirectory(prefix='large-token-evidence-preflight-') as directory:
            folder = Path(directory); cfg = {**cfg, 'seed': 0}; torch.manual_seed(20261124); torch.cuda.manual_seed_all(20261124)
            model = RegisterClassifierContextModel(cfg, dims, len(source['diseases']), geometry, folder/'training_input_witness.npz').cuda().train()
            local = torch.randperm(len(tf), device='cuda')[:256].cpu().numpy(); gx = {key: torch.from_numpy(value[fit][local]).cuda() for key, value in normalized.items()}
            gy = torch.tensor(tf.iloc[local].label.to_numpy(), device='cuda'); gd = torch.tensor(tf.iloc[local].disease.map(mapping).to_numpy(), device='cuda')
            gw = {'leaf': torch.from_numpy(sample_weights(tf, tf.label)[local]).cuda(), 'patient_ids': torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[local]).cuda()}
            loss = token_evidence_loss(model(gx), gy, gd, gw, cfg); loss.backward()
            c = {'cfg': cfg, 'dims': dims, 'diseases': source['diseases'], 'stats': stats, 'state_dict': {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}}
            witnesses.append({'kind': kind, 'witness': audit_training_witness(c, folder, features, frame, fit)})
            del model; torch.cuda.empty_cache()
    assert len(set(counts)) == 1
    dump(ROOT/'verification_register_classifier_context.json', {'passed': True, 'manifest_sha256': digest, 'cases': cases, 'base_normal_geometry': base_diagnostic,
        'side_normal_geometries': diagnostics, 'actual_cuda_full_task_witnesses': witnesses,
        'code_sha256': {name: sha(ROOT/name) for name in ['register_classifier_context.py', 'audit_register_classifier_context.py', 'verify_register_classifier_context.py', 'register_multiquery_data.py']},
        'checks': ['Four actual tokenstreams256image complete task/allparametergradients independently reconstructed',
            'OriginalnormalL initialization/RNG/fulltask/basegradients identical atzeroreadout; equalparameters across all four attentionarms; dotcontrol exact M8trainedidentity',
            'Fitnormalpatient tokenPCA recovered by Gram spectrum; independent projections/query/keys/values/attention/delta',
            'Tokenpermutation invariance, joint classifierrow/map permutation invariance, classbalanced diseasequerypooling, linkedclassifiercontext gradient and detachedcontext absence, nonzero hiddenquery connectivity, optimizerupdates/frozenbuffers/strictrestore and fouractualCUDA firsttask/allgradientreplays'],
        'classification_fits': 0, 'classification_scores': 0, 'external_test_evaluated': False})
    print('Four register classifier-context full-task gradient cases and four actual CUDA witnesses passed.', flush=True)


if __name__ == '__main__': main()
