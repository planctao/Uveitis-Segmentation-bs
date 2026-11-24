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

from large_token_evidence import LargeTokenEvidenceModel, fit_side_geometry, token_evidence_loss, DETAILS
from register_position_calibration import RegisterPositionCalibrationModel, fit_position_moments
from normal_subspace import independent_transform
from audit_register_position_calibration import ordinary, independent_forward, independent_side_geometry, audit_training_witness, independent_position_moments
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
    for arm in json.loads((ROOT/'results/next_register_position_calibration_hypothesis.json').read_text())['configuration_plan']:
        stream = 'registers'
        features = {**base_features, 'side_tokens': token_bank[stream]}; normalized, stats = normalize(features, mask)
        geometry = {**base_geometry, **fit_side_geometry(tf, normalized['side_tokens'][fit])}
        diagnostics[arm['name']] = independent_side_geometry(tf, normalized['side_tokens'][fit], geometry)
        projected = independent_transform(normalized['side_tokens'][fit], geometry['side_center'], geometry['side_basis'], 1.)
        scope = 'all' if arm['kind'] == 'all_standardize' else 'normal'
        moments = fit_position_moments(tf, projected, scope)
        diagnostics[arm['name']]['moments'] = independent_position_moments(tf, projected, moments, scope)
        geometry.update(moments)
        raw = {key: value[fit][ids] for key, value in normalized.items()}; x = {key: torch.from_numpy(value) for key, value in raw.items()}
        cfg = {**source['cfg'], 'token_evidence_stream': stream, 'register_position_kind': arm['kind']}
        torch.manual_seed(20285202); base = NormalSuppressedModel(cfg, source['dims'], len(source['diseases']), base_geometry); expected_rng = torch.random.get_rng_state().clone()
        torch.manual_seed(20285202); model = RegisterPositionCalibrationModel(cfg, dims, len(source['diseases']), geometry)
        assert torch.equal(expected_rng, torch.random.get_rng_state())
        for name, value in base.state_dict().items(): assert torch.equal(value, model.state_dict()[name])
        rng = torch.random.get_rng_state().clone(); out = model(x); actual = loss_fn(out, y, disease, weights, cfg); actual.backward(); after = torch.random.get_rng_state().clone()
        torch.random.set_rng_state(rng); original = loss_fn(base({key: x[key] for key in KEYS}), y, disease, weights, cfg); original.backward()
        assert torch.equal(actual, original) and torch.equal(after, torch.random.get_rng_state()) and not torch.count_nonzero(out['token_delta'])
        zero_errors = []
        for name, p in base.named_parameters():
            other = dict(model.named_parameters())[name].grad; error = float((p.grad-other).abs().max()); zero_errors.append(error)
            assert torch.allclose(p.grad, other, atol=1e-7, rtol=1e-5), (arm['name'], name, error)
        print({'arm': arm['name'], 'maximum_zero_residual_base_gradient_error': max(zero_errors)}, flush=True)
        assert model.token_evidence.readout.weight.grad.abs().sum() > 0
        state = model.state_dict(); state.update(source['state_dict']); model.load_state_dict(state, strict=True)
        identity_error = None
        if arm['kind'] == 'none':
            trained = torch.load(ROOT/'runs/large_token_evidence/registers/validation/seed_0/model.pt', map_location='cpu', weights_only=False)
            state = model.state_dict(); state.update(trained['state_dict']); model.load_state_dict(state, strict=True); model.eval()
            single = LargeTokenEvidenceModel(trained['cfg'], trained['dims'], len(trained['diseases'])).eval(); single.load_state_dict(trained['state_dict'], strict=True)
            with torch.no_grad():
                identity_error = float((single(x)['p']-model(x)['p']).abs().max()); assert identity_error == 0
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
        # Scale floor=.05 can amplify side-stream projection roundoff twentyfold.
        intermediate_errors = {}
        for key in ['p']+DETAILS:
            error = float((out[key]-manual_out[key]).abs().max())
            intermediate_errors[key] = error
            side_key = key in ['token_projected', 'token_keys', 'token_values', 'token_attention', 'token_pooled']
            tolerance = 1.4e-4 if side_key and arm['kind'] in ['normal_standardize', 'all_standardize'] else 7e-6
            assert torch.allclose(out[key], manual_out[key], atol=tolerance, rtol=1e-4), (key, error)
        model.eval()
        with torch.no_grad():
            expected = model(x)['p']; permutation = torch.arange(x['side_tokens'].shape[1]-1, -1, -1)
            saved_mean = model.position_mean.clone(); saved_scale = model.position_scale.clone()
            model.position_mean.copy_(saved_mean[permutation]); model.position_scale.copy_(saved_scale[permutation])
            changed = model({**x, 'side_tokens': x['side_tokens'][:, permutation]})['p']; permutation_error = float((changed-expected).abs().max()); assert permutation_error < 4e-6
            model.position_mean.copy_(saved_mean); model.position_scale.copy_(saved_scale)
            tokens = model._tokens.clone(); hidden = torch.randn(256, 256)
            a = model.token_evidence(tokens, hidden); b = model.token_evidence(tokens, hidden.roll(1, 0))
            query_change = float((a['token_attention']-b['token_attention']).abs().max()); assert query_change > 1e-6
        frozen = {key: value.clone() for key, value in model.named_buffers()}; before = model.token_evidence.key.weight.detach().clone()
        torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.1).step(); assert not torch.equal(before, model.token_evidence.key.weight)
        for name, value in frozen.items(): assert torch.equal(value, dict(model.named_buffers())[name])
        restored = RegisterPositionCalibrationModel(cfg, dims, len(source['diseases'])).eval(); restored.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad(): assert torch.equal(model(x)['p'], restored(x)['p'])
        counts.append(sum(p.numel() for p in model.parameters())); cases.append({'arm': arm, 'original_M8_shared_control_error': identity_error, 'maximum_zero_residual_base_gradient_error': max(zero_errors), 'independent_intermediate_errors': intermediate_errors, 'batch_size': 256, 'tokens': x['side_tokens'].shape[1],
            'parameters': counts[-1], 'maximum_complete_task_gradient_error': max(errors), 'token_permutation_error': permutation_error, 'hidden_query_attention_change': query_change})
        with tempfile.TemporaryDirectory(prefix='large-token-evidence-preflight-') as directory:
            folder = Path(directory); cfg = {**cfg, 'seed': 0}; torch.manual_seed(20261124); torch.cuda.manual_seed_all(20261124)
            model = RegisterPositionCalibrationModel(cfg, dims, len(source['diseases']), geometry, folder/'training_input_witness.npz').cuda().train()
            local = torch.randperm(len(tf), device='cuda')[:256].cpu().numpy(); gx = {key: torch.from_numpy(value[fit][local]).cuda() for key, value in normalized.items()}
            gy = torch.tensor(tf.iloc[local].label.to_numpy(), device='cuda'); gd = torch.tensor(tf.iloc[local].disease.map(mapping).to_numpy(), device='cuda')
            gw = {'leaf': torch.from_numpy(sample_weights(tf, tf.label)[local]).cuda(), 'patient_ids': torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[local]).cuda()}
            loss = token_evidence_loss(model(gx), gy, gd, gw, cfg); loss.backward()
            c = {'cfg': cfg, 'dims': dims, 'diseases': source['diseases'], 'stats': stats, 'state_dict': {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}}
            witnesses.append({'arm': arm, 'witness': audit_training_witness(c, folder, features, frame, fit)})
            del model; torch.cuda.empty_cache()
    assert len(set(counts)) == 1
    dump(ROOT/'verification_register_position_calibration.json', {'passed': True, 'manifest_sha256': digest, 'cases': cases, 'base_normal_geometry': base_diagnostic,
        'side_normal_geometries': diagnostics, 'actual_cuda_full_task_witnesses': witnesses,
        'code_sha256': {name: sha(ROOT/name) for name in ['register_position_calibration.py', 'audit_register_position_calibration.py', 'verify_register_position_calibration.py', 'large_token_data.py']},
        'checks': ['Four actual tokenstreams256image complete task/allparametergradients independently reconstructed',
            'OriginalnormalL initialization/RNG/task identical atzero; basegradients numericallymatched (atol1e-7,rtol1e-5); equalparameters/allfourregisters; sharedcontrol exactM8learnedidentity',
            'Fitnormalpatient tokenPCA recovered by Gram spectrum; independent projections/query/keys/values/attention/delta',
            'Jointtoken/momentpermutation invariance, nonzero hiddenquery connectivity, optimizerupdates/frozenbuffers/strictrestore and fouractualCUDA firsttask/allgradientreplays'],
        'side_intermediate_atol_standardization': .00014, 'probability_atol': 7e-6,
        'roundoff_note': 'Sidecalibration scale>=.05 may amplify projection roundoff20x. Fulltask/gradient/probability tolerances unchanged; allperstage errors recorded beforeclassification.',
        'classification_fits': 0, 'classification_scores': 0, 'external_test_evaluated': False})
    print('Four register position-calibration tasks, patient moments and CUDA witnesses passed.', flush=True)


if __name__ == '__main__': main()
