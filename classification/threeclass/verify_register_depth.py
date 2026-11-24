"""Real depth features, complete-task gradients and matched repeated controls."""
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment import ROOT, KEYS, normalize, loss_fn, sample_weights, dump
from normal_subspace import NormalSuppressedModel, fit_normal_geometry
from register_depth_data import load_all
from register_depth_features import TOKEN_KEYS
from register_depth import RegisterDepthModel, DETAILS, register_depth_loss
from audit_register_depth import ordinary, independent_forward, audit_training_witness
from large_token_evidence import LargeTokenEvidenceModel, fit_side_geometry
from audit_large_token_evidence import independent_side_geometry
from verify_normal_subspace import independent_geometry
from audit_normal_tangent import complete_task
from kernel_residual_experiments import sha


def main():
    torch.set_num_threads(4); frame, features, digest = load_all()
    fit = np.flatnonzero(frame.split.eq('train')); mask = np.zeros(len(frame), bool); mask[fit] = True
    tf = frame.iloc[fit].reset_index(drop=True); normalized, stats = normalize(features, mask)
    geometry = fit_normal_geometry(tf, {key: normalized[key][fit] for key in KEYS})
    diagnostics = {'base': independent_geometry(tf, {key: normalized[key][fit] for key in KEYS}, geometry)}
    for key in TOKEN_KEYS:
        side = fit_side_geometry(tf, normalized[key][fit]); diagnostics[key] = independent_side_geometry(tf, normalized[key][fit], side)
        geometry.update({key+'_'+suffix: side['side_'+suffix] for suffix in ['center', 'basis', 'eigenvalues', 'normal_patients']})
    prior = dict(np.load(ROOT/'runs/large_token_evidence/geometry/validation/geometry.npz'))
    for suffix in ['center', 'basis', 'eigenvalues', 'normal_patients']:
        assert np.array_equal(geometry['final_registers_'+suffix], prior['registers_side_'+suffix])
    source = torch.load(ROOT/'runs/normal_large_reference/large_dual/validation/seed_0/model.pt', map_location='cpu', weights_only=False)
    token_source = torch.load(ROOT/'runs/large_token_evidence/registers/validation/seed_0/model.pt', map_location='cpu', weights_only=False)
    dims = {**source['dims'], **{key: 2048 for key in TOKEN_KEYS}}
    positions = np.random.default_rng(20285601).choice(len(tf), 256, replace=False)
    raw = {key: value[fit][positions] for key, value in normalized.items()}; x = {key: torch.from_numpy(value) for key, value in raw.items()}
    mapping = {name: i for i, name in enumerate(source['diseases'])}
    y = torch.tensor(tf.iloc[positions].label.to_numpy()); disease = torch.tensor(tf.iloc[positions].disease.map(mapping).to_numpy())
    weights = {'leaf': torch.from_numpy(sample_weights(tf, tf.label)[positions]), 'patient_ids': torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[positions])}
    hypothesis = json.loads((ROOT/'results/next_register_depth_hypothesis.json').read_text())
    cases, witnesses, counts = [], [], []
    for arm in hypothesis['configuration_plan']:
        cfg = {**source['cfg'], 'register_depth_kind': arm['kind']}
        torch.manual_seed(20285602); base = NormalSuppressedModel(cfg, source['dims'], len(source['diseases']), geometry); expected_rng = torch.random.get_rng_state().clone()
        torch.manual_seed(20285602); model = RegisterDepthModel(cfg, dims, len(source['diseases']), geometry)
        assert torch.equal(expected_rng, torch.random.get_rng_state())
        for name, value in base.state_dict().items(): assert torch.equal(value, model.state_dict()[name])
        rng = torch.random.get_rng_state().clone(); out = model(x); actual = loss_fn(out, y, disease, weights, cfg); actual.backward(); after = torch.random.get_rng_state().clone()
        torch.random.set_rng_state(rng); original = loss_fn(base({key: x[key] for key in KEYS}), y, disease, weights, cfg); original.backward()
        assert torch.equal(actual, original) and torch.equal(after, torch.random.get_rng_state()) and not torch.count_nonzero(out['token_delta'])
        for name, parameter in base.named_parameters(): assert torch.equal(parameter.grad, dict(model.named_parameters())[name].grad), name
        assert model.token_evidence.readout.weight.grad.abs().sum() > 0
        state = model.state_dict(); state.update({key: value for key, value in token_source['state_dict'].items() if not key.startswith('side_normal_')})
        model.load_state_dict(state, strict=True); model.zero_grad()
        if arm['kind'] in ['final', 'middle']:
            key = 'final_registers' if arm['kind'] == 'final' else 'middle_registers'
            side_geometry = {**geometry, **{'side_'+suffix: geometry[key+'_'+suffix] for suffix in ['center', 'basis', 'eigenvalues', 'normal_patients']}}
            single = LargeTokenEvidenceModel(token_source['cfg'], token_source['dims'], len(source['diseases']), side_geometry).eval()
            single_state = single.state_dict(); single_state.update({name: value for name, value in model.state_dict().items() if name in single_state and not name.startswith('side_normal_')}); single.load_state_dict(single_state, strict=True)
            model.eval()
            with torch.no_grad():
                repeated = model(x)['p']; simple = single({**{key: x[key] for key in KEYS}, 'side_tokens': x[key]})['p']
                duplicate_error = float((repeated-simple).abs().max()); assert duplicate_error < 2e-6
            model.train()
        else: duplicate_error = None
        state = model.state_dict(keep_vars=True)
        independent = {key: value.detach().clone().requires_grad_(key.startswith('token_evidence.')) for key, value in state.items()}
        plain = ordinary({**source, 'cfg': cfg, 'dims': dims, 'state_dict': state}).train()
        rng = torch.random.get_rng_state().clone(); out = model(x); actual = register_depth_loss(out, y, disease, weights, cfg); actual.backward(); after = torch.random.get_rng_state().clone()
        torch.random.set_rng_state(rng); manual_out = independent_forward(plain, raw, independent, cfg); manual = complete_task(manual_out, y, disease, weights, cfg); manual.backward()
        assert torch.equal(after, torch.random.get_rng_state()) and torch.allclose(actual, manual, atol=6e-6, rtol=6e-5)
        errors = []
        for name, parameter in model.named_parameters():
            other = independent[name].grad if name.startswith('token_evidence.') else dict(plain.named_parameters())[name].grad
            assert parameter.grad is not None and other is not None and torch.allclose(parameter.grad, other, atol=1e-5, rtol=2e-3), name
            errors.append(float((parameter.grad-other).abs().max()))
        for key in ['p']+DETAILS: assert torch.allclose(out[key], manual_out[key], atol=7e-6, rtol=1e-4), key
        assert out['depth_tokens'].shape == (256, 12, 2048) and torch.isfinite(out['depth_tokens']).all()
        model.eval()
        with torch.no_grad():
            expected = model(x)['p']; permuted = model({**x, **{key: x[key].flip(1) for key in TOKEN_KEYS}})['p']
            permutation_error = float((permuted-expected).abs().max()); assert permutation_error < 4e-6
        frozen = {name: value.clone() for name, value in model.named_buffers()}; before = model.token_evidence.key.weight.detach().clone()
        torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.1).step(); assert not torch.equal(before, model.token_evidence.key.weight)
        for name, value in frozen.items(): assert torch.equal(value, dict(model.named_buffers())[name])
        restored = RegisterDepthModel(cfg, dims, len(source['diseases'])).eval(); restored.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad(): assert torch.equal(model(x)['p'], restored(x)['p'])
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
        cases.append({'arm': arm, 'batch_size': 256, 'tokens': 12, 'parameters': counts[-1],
            'maximum_complete_task_gradient_error': max(errors), 'repeated_vs_single_control_error': duplicate_error, 'joint_token_permutation_error': permutation_error})
        with tempfile.TemporaryDirectory(prefix='register-depth-preflight-') as directory:
            folder = Path(directory); cfg = {**cfg, 'seed': 0}; torch.manual_seed(20261124); torch.cuda.manual_seed_all(20261124)
            gm = RegisterDepthModel(cfg, dims, len(source['diseases']), geometry, folder/'training_input_witness.npz').cuda().train()
            local = torch.randperm(len(tf), device='cuda')[:256].cpu().numpy()
            gx = {key: torch.from_numpy(value[fit][local]).cuda() for key, value in normalized.items()}
            gy = torch.tensor(tf.iloc[local].label.to_numpy(), device='cuda'); gd = torch.tensor(tf.iloc[local].disease.map(mapping).to_numpy(), device='cuda')
            gw = {'leaf': torch.from_numpy(sample_weights(tf, tf.label)[local]).cuda(), 'patient_ids': torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[local]).cuda()}
            loss = register_depth_loss(gm(gx), gy, gd, gw, cfg); loss.backward()
            c = {'cfg': cfg, 'dims': dims, 'diseases': source['diseases'], 'stats': stats, 'state_dict': {key: value.detach().cpu().clone() for key, value in gm.state_dict().items()}}
            witnesses.append({'arm': arm, 'witness': audit_training_witness(c, folder, features, frame, fit)})
            del gm; torch.cuda.empty_cache()
    assert len(set(counts)) == 1
    dump(ROOT/'verification_register_depth.json', {'passed': True, 'manifest_sha256': digest, 'cases': cases,
        'normal_geometry': diagnostics, 'actual_cuda_full_task_witnesses': witnesses,
        'code_sha256': {name: sha(ROOT/name) for name in ['register_depth.py', 'audit_register_depth.py', 'verify_register_depth.py', 'register_depth_data.py']},
        'checks': ['Four actual256image tasks/allgradients independently restored with trainednonzeroreadout',
            'OriginalnormalL init/RNG/task/basegradients identical atzero; equalparameters/all12tokens',
            'Independentthree normalspaces/depthdifferences/querypool; repeatedfinal/middle controls match single-streamlearnedweights',
            'Jointregisterpermutation, updates/frozenbuffers/strictrestore andfouractualCUDA firsttasks/allgradients'],
        'classification_fits': 0, 'classification_scores': 0, 'external_test_evaluated': False})
    print('Four register-depth full-task cases/CUDA witnesses and single-stream control identities passed.', flush=True)


if __name__ == '__main__': main()
