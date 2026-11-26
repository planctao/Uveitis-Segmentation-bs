"""Real feature full-task gradients, identical normal-L base and CUDA input replay."""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment import ROOT, KEYS, normalize, loss_fn, sample_weights, dump
from normal_subspace import NormalSuppressedModel, fit_normal_geometry
from large_token_data import load_all
from large_token_evidence import LargeTokenEvidenceModel, TokenQueryResidual, fit_side_geometry
from register_recurrent_refinement import RegisterRecurrentRefinementModel, KINDS, TRACE, DETAILS
from compact_token_witness import token_evidence_loss, verify_legacy_equivalence
from audit_register_recurrent_refinement_compact import ordinary, independent_forward, independent_side_geometry, audit_training_witness
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
        cfg = {**source['cfg'], 'token_evidence_stream': stream, 'register_recurrent_refinement': kind}
        torch.manual_seed(20285202); base = NormalSuppressedModel(cfg, source['dims'], len(source['diseases']), base_geometry); expected_rng = torch.random.get_rng_state().clone()
        torch.manual_seed(20285202); model = RegisterRecurrentRefinementModel(cfg, dims, len(source['diseases']), geometry)
        assert torch.equal(expected_rng, torch.random.get_rng_state())
        for name, value in base.state_dict().items(): assert torch.equal(value, model.state_dict()[name])
        rng = torch.random.get_rng_state().clone(); out = model(x); actual = loss_fn(out, y, disease, weights, cfg); actual.backward(); after = torch.random.get_rng_state().clone()
        torch.random.set_rng_state(rng); original = loss_fn(base({key: x[key] for key in KEYS}), y, disease, weights, cfg); original.backward()
        assert torch.equal(actual, original) and torch.equal(after, torch.random.get_rng_state()) and not torch.count_nonzero(out['token_delta'])
        for name, p in base.named_parameters(): assert torch.equal(p.grad, dict(model.named_parameters())[name].grad), name
        assert model.token_evidence.readout.weight.grad.abs().sum() > 0
        with torch.random.fork_rng(devices=[]):
            reference_model = LargeTokenEvidenceModel(cfg, dims, len(source['diseases']), geometry)
            reference_model.load_state_dict(model.state_dict(),strict=True)
            torch.random.set_rng_state(rng)
            reference_loss = loss_fn(reference_model(x), y, disease, weights, cfg); reference_loss.backward()
        initial_gradient_errors = []
        for name, parameter in model.named_parameters():
            other = dict(reference_model.named_parameters())[name].grad
            assert parameter.grad is not None and other is not None and torch.allclose(parameter.grad,other,atol=1e-6,rtol=3e-3),name
            initial_gradient_errors.append(float((parameter.grad-other).abs().max()))
        assert torch.equal(actual,reference_loss)
        state = model.state_dict(); state.update(source['state_dict']); model.load_state_dict(state, strict=True)
        control_error = None
        if kind == 'control':
            trained = torch.load(ROOT/'runs/large_token_evidence/registers/validation/seed_0/model.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(trained['state_dict'],strict=True); model.eval()
            original_model = LargeTokenEvidenceModel(trained['cfg'],trained['dims'],len(trained['diseases'])).eval()
            original_model.load_state_dict(trained['state_dict'],strict=True)
            with torch.no_grad():
                control_error=float((model(x)['p']-original_model(x)['p']).abs().max()); assert control_error==0.
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
        for key in ['p']+DETAILS+TRACE: assert torch.allclose(out[key], manual_out[key], atol=7e-6, rtol=1e-4), key
        model.eval()
        with torch.no_grad():
            expected = model(x)['p']; permutation = torch.arange(x['side_tokens'].shape[1]-1, -1, -1)
            changed = model({**x, 'side_tokens': x['side_tokens'][:, permutation]})['p']; permutation_error = float((changed-expected).abs().max()); assert permutation_error < 4e-6
            tokens = model._tokens.clone(); hidden = torch.randn(256, 256)
            a = model.token_evidence(tokens, hidden); b = model.token_evidence(tokens, hidden.roll(1, 0))
            query_change = float((a['token_attention']-b['token_attention']).abs().max()); assert query_change > 1e-6
            expected_evaluations = {'control':1,'euler2':2,'euler4':4,'heun':2}[kind]
            assert a['refinement_hidden'].shape == (256,expected_evaluations,256)
            assert a['refinement_attention'].shape == (256,expected_evaluations,4)
            first = TokenQueryResidual.forward(model.token_evidence,tokens,hidden)
            if kind == 'heun':
                assert torch.allclose(a['refinement_hidden'][:,1],hidden+first['token_delta'],atol=2e-6,rtol=1e-5)
                expected_hidden=hidden+a['refinement_delta'].mean(1)
            elif kind == 'control':expected_hidden=hidden+first['token_delta']
            else:
                for step in range(1,expected_evaluations):
                    assert torch.allclose(a['refinement_hidden'][:,step],a['refinement_hidden'][:,step-1]+a['refinement_delta'][:,step-1]/expected_evaluations,atol=2e-6,rtol=1e-5)
                expected_hidden=a['refinement_hidden'][:,-1]+a['refinement_delta'][:,-1]/expected_evaluations
            assert torch.allclose(a['refinement_updated_hidden'],expected_hidden,atol=2e-6,rtol=1e-5)
            feedback_attention_change=0. if kind=='control' else float((a['refinement_attention'][:,-1]-a['refinement_attention'][:,0]).abs().max())
            if kind!='control': assert feedback_attention_change > 1e-8

        frozen = {key: value.clone() for key, value in model.named_buffers()}; before = model.token_evidence.key.weight.detach().clone()
        torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.1).step(); assert not torch.equal(before, model.token_evidence.key.weight)
        for name, value in frozen.items(): assert torch.equal(value, dict(model.named_buffers())[name])
        restored = RegisterRecurrentRefinementModel(cfg, dims, len(source['diseases'])).eval(); restored.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad(): assert torch.equal(model(x)['p'], restored(x)['p'])
        counts.append(sum(p.numel() for p in model.parameters())); cases.append({'kind':kind,'expected_field_evaluations':expected_evaluations,'m8_exact_control_error':control_error,'initial_m8_allgradient_error':max(initial_gradient_errors),'query_feedback_attention_change':feedback_attention_change,'stream': stream, 'batch_size': 256, 'tokens': x['side_tokens'].shape[1],
            'parameters': counts[-1], 'maximum_complete_task_gradient_error': max(errors), 'token_permutation_error': permutation_error, 'hidden_query_attention_change': query_change})
        with tempfile.TemporaryDirectory(prefix='large-token-evidence-preflight-') as directory:
            folder = Path(directory); cfg = {**cfg, 'seed': 0}; torch.manual_seed(20261124); torch.cuda.manual_seed_all(20261124)
            model = RegisterRecurrentRefinementModel(cfg, dims, len(source['diseases']), geometry, folder/'training_input_witness.npz').cuda().train()
            local = torch.randperm(len(tf), device='cuda')[:256].cpu().numpy(); gx = {key: torch.from_numpy(value[fit][local]).cuda() for key, value in normalized.items()}
            gy = torch.tensor(tf.iloc[local].label.to_numpy(), device='cuda'); gd = torch.tensor(tf.iloc[local].disease.map(mapping).to_numpy(), device='cuda')
            gw = {'leaf': torch.from_numpy(sample_weights(tf, tf.label)[local]).cuda(), 'patient_ids': torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[local]).cuda()}
            loss = token_evidence_loss(model(gx), gy, gd, gw, cfg); loss.backward()
            compact_bytes=(folder/'training_input_witness.npz').stat().st_size
            if kind=='control':
                verify_legacy_equivalence(folder/'training_input_witness.npz',ROOT/'runs/large_token_evidence/registers/validation/seed_0/training_input_witness.npz')

            c = {'cfg': cfg, 'dims': dims, 'diseases': source['diseases'], 'stats': stats, 'state_dict': {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}}
            witnesses.append({'kind':kind,'compact_witness_bytes':compact_bytes,'stream': stream, 'witness': audit_training_witness(c, folder, features, frame, fit)})
            del model; torch.cuda.empty_cache()
    assert len(set(counts)) == 1
    dump(ROOT/'verification_register_recurrent_refinement_compact.json', {'passed': True, 'manifest_sha256': digest, 'cases': cases, 'base_normal_geometry': base_diagnostic,
        'side_normal_geometries': diagnostics, 'actual_cuda_full_task_witnesses': witnesses,
        'code_sha256': {name: sha(ROOT/name) for name in ['register_recurrent_refinement.py','audit_register_recurrent_refinement.py','audit_register_recurrent_refinement_compact.py','compact_token_witness.py','verify_register_recurrent_refinement_compact.py','large_token_data.py']},
        'checks': ['Four actual256image complete tasks/allparametergradients and allnonzero readoutfeedbacksteps independently reconstructed',
            'OriginalnormalL initialization/RNG/zeroreadout fulltask/basegradients identical; allparametergradients match singlepassM8 atdeclaredexisting tolerance; equalparameters and originalcontrol exacttrainedM8',
            'Fitnormalpatient tokenPCA recovered by Gram spectrum; independent projections/query/keys/values/attention/delta',
            'Intermediatehidden/attention traces and evaluationcounts verified; fixedtotalstep Euler/Heun updates independentlyunrolled; nonzero queryfeedback, tokenpermutation, optimizerupdates/frozenbuffers/strictrestore and fouractualCUDA firsttask/allgradientreplays'],
        'classification_fits':0,'classification_scores':0,'external_test_evaluated': False})
    print('Four compact register recurrent-refinement full-task gradient cases and four actual CUDA witnesses passed.', flush=True)


if __name__ == '__main__': main()
