"""Real 256-image task/gradient preflight for four bounded-interaction arms."""
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment import ROOT, KEYS, normalize, loss_fn, sample_weights, dump
from normal_subspace import NormalSuppressedModel
from large_token_data import load_all
from large_token_evidence_experiments import geometry_for
from audit_large_token_evidence import independent_side_geometry
from verify_normal_subspace import independent_geometry
from large_token_interaction import LargeTokenInteractionModel, STREAMS, DETAILS, token_pair_loss
from audit_large_token_interaction import ordinary, independent_forward, audit_training_witness
from audit_normal_tangent import complete_task
from kernel_residual_experiments import sha


def main():
    torch.set_num_threads(4); frame, base_features, tokens, digest = load_all()
    fit = np.flatnonzero(frame.split.eq('train')); mask = np.zeros(len(frame), bool); mask[fit] = True
    tf = frame.iloc[fit].reset_index(drop=True)
    features = {**base_features, 'mid_tokens': tokens['mid_regions'], 'register_tokens': tokens['registers']}; del tokens
    normalized, stats = normalize(features, mask)
    saved = dict(np.load(ROOT/'runs/large_token_evidence/geometry/validation/geometry.npz'))
    geometry = {key: saved[key] for key in ['normal_patients']+[name+'_'+suffix for name in KEYS for suffix in ['center', 'basis', 'eigenvalues']]}
    diagnostics = {'base': independent_geometry(tf, {key: normalized[key][fit] for key in KEYS}, geometry)}
    for name, stream in zip(STREAMS, ['mid_regions', 'registers']):
        side = geometry_for(saved, stream); diagnostics[name] = independent_side_geometry(tf, normalized[name][fit], side)
        geometry.update({name+'_'+suffix: side['side_'+suffix] for suffix in ['center', 'basis', 'eigenvalues', 'normal_patients']})
    source = torch.load(ROOT/'runs/normal_large_reference/large_dual/validation/seed_0/model.pt', map_location='cpu', weights_only=False)
    dims = {**source['dims'], **{name: 2048 for name in STREAMS}}
    local = np.random.default_rng(20285301).choice(len(tf), 256, replace=False)
    raw = {key: value[fit][local] for key, value in normalized.items()}; x = {key: torch.from_numpy(value) for key, value in raw.items()}
    mapping = {name: i for i, name in enumerate(source['diseases'])}
    y = torch.tensor(tf.iloc[local].label.to_numpy()); disease = torch.tensor(tf.iloc[local].disease.map(mapping).to_numpy())
    weights = {'leaf': torch.from_numpy(sample_weights(tf, tf.label)[local]), 'patient_ids': torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[local])}
    hypothesis = json.loads((ROOT/'results/next_large_token_interaction_hypothesis.json').read_text())
    cases, witnesses, counts = [], [], {}
    for arm in hypothesis['configuration_plan']:
        cfg = {**source['cfg'], 'token_pair_rank': arm['rank'], 'token_pair_kind': arm['kind']}
        torch.manual_seed(20285302); base = NormalSuppressedModel(cfg, source['dims'], len(source['diseases']), geometry); expected_rng = torch.random.get_rng_state().clone()
        torch.manual_seed(20285302); model = LargeTokenInteractionModel(cfg, dims, len(source['diseases']), geometry)
        assert torch.equal(expected_rng, torch.random.get_rng_state())
        for name, value in base.state_dict().items(): assert torch.equal(value, model.state_dict()[name])
        rng = torch.random.get_rng_state().clone(); out = model(x); actual = loss_fn(out, y, disease, weights, cfg); actual.backward(); after = torch.random.get_rng_state().clone()
        torch.random.set_rng_state(rng); original = loss_fn(base({key: x[key] for key in KEYS}), y, disease, weights, cfg); original.backward()
        assert torch.equal(actual, original) and torch.equal(after, torch.random.get_rng_state()) and not torch.count_nonzero(out['pair_delta'])
        for name, parameter in base.named_parameters(): assert torch.equal(parameter.grad, dict(model.named_parameters())[name].grad), name
        assert model.token_pair.readout.weight.grad.abs().sum() > 0
        state = model.state_dict(); state.update(source['state_dict']); model.load_state_dict(state, strict=True)
        with torch.no_grad():
            readout = model.token_pair.readout
            readout.weight.copy_(torch.sin(torch.arange(readout.weight.numel()).reshape(readout.weight.shape)*.17)*.01)
            readout.bias.copy_(torch.linspace(-.01, .01, 256))
        model.zero_grad(); state = model.state_dict(keep_vars=True)
        independent = {key: value.detach().clone().requires_grad_(key.startswith('token_pair.')) for key, value in state.items()}
        plain = ordinary({**source, 'cfg': cfg, 'dims': dims, 'state_dict': state}).train()
        rng = torch.random.get_rng_state().clone(); out = model(x); actual = token_pair_loss(out, y, disease, weights, cfg); actual.backward(); after = torch.random.get_rng_state().clone()
        torch.random.set_rng_state(rng); manual_out = independent_forward(plain, raw, independent, cfg); manual = complete_task(manual_out, y, disease, weights, cfg); manual.backward()
        assert torch.equal(after, torch.random.get_rng_state()) and torch.allclose(actual, manual, atol=6e-6, rtol=6e-5)
        errors = []
        for name, parameter in model.named_parameters():
            other = independent[name].grad if name.startswith('token_pair.') else dict(plain.named_parameters())[name].grad
            assert parameter.grad is not None and other is not None and torch.allclose(parameter.grad, other, atol=1e-5, rtol=2e-3), name
            errors.append(float((parameter.grad-other).abs().max()))
        for key in ['p']+DETAILS: assert torch.allclose(out[key], manual_out[key], atol=7e-6, rtol=1e-4), key
        for name in STREAMS: assert out[name+'_bounded'].abs().max() <= 1
        if arm['kind'] == 'product': assert out['pair_joint'].abs().max() <= 1
        model.eval()
        with torch.no_grad():
            expected = model(x)['p']; permutation_errors = {}
            for name in STREAMS:
                changed = model({**x, name: x[name].flip(1)})['p']; error = float((changed-expected).abs().max()); assert error < 4e-6; permutation_errors[name] = error
            hidden = torch.randn(256, 256); streams = {name: model._streams[name].clone() for name in STREAMS}
            attention_changes = {}
            a = model.token_pair(streams, hidden); b = model.token_pair(streams, hidden.roll(1, 0))
            for name in STREAMS:
                change = float((a[name+'_attention']-b[name+'_attention']).abs().max()); assert change > 1e-6; attention_changes[name] = change
        frozen = {key: value.clone() for key, value in model.named_buffers()}
        before = {name: model.token_pair.keys[name].weight.detach().clone() for name in STREAMS}
        torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.1).step()
        for name in STREAMS: assert not torch.equal(before[name], model.token_pair.keys[name].weight)
        for name, value in frozen.items(): assert torch.equal(value, dict(model.named_buffers())[name])
        restored = LargeTokenInteractionModel(cfg, dims, len(source['diseases'])).eval(); restored.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad(): assert torch.equal(model(x)['p'], restored(x)['p'])
        count = sum(p.numel() for p in model.parameters()); counts.setdefault(arm['rank'], []).append(count)
        cases.append({'arm': arm, 'batch_size': 256, 'parameters': count, 'maximum_complete_task_gradient_error': max(errors),
            'permutation_errors': permutation_errors, 'hidden_query_attention_changes': attention_changes})
        with tempfile.TemporaryDirectory(prefix='large-token-interaction-preflight-') as directory:
            folder = Path(directory); cfg = {**cfg, 'seed': 0}; torch.manual_seed(20261124); torch.cuda.manual_seed_all(20261124)
            gm = LargeTokenInteractionModel(cfg, dims, len(source['diseases']), geometry, folder/'training_input_witness.npz').cuda().train()
            positions = torch.randperm(len(tf), device='cuda')[:256].cpu().numpy()
            gx = {key: torch.from_numpy(value[fit][positions]).cuda() for key, value in normalized.items()}
            gy = torch.tensor(tf.iloc[positions].label.to_numpy(), device='cuda'); gd = torch.tensor(tf.iloc[positions].disease.map(mapping).to_numpy(), device='cuda')
            gw = {'leaf': torch.from_numpy(sample_weights(tf, tf.label)[positions]).cuda(), 'patient_ids': torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[positions]).cuda()}
            loss = token_pair_loss(gm(gx), gy, gd, gw, cfg); loss.backward()
            c = {'cfg': cfg, 'dims': dims, 'diseases': source['diseases'], 'stats': stats, 'state_dict': {key: value.detach().cpu().clone() for key, value in gm.state_dict().items()}}
            witnesses.append({'arm': arm, 'witness': audit_training_witness(c, folder, features, frame, fit)})
            del gm; torch.cuda.empty_cache()
    assert all(len(v) == 2 and v[0] == v[1] for v in counts.values())
    dump(ROOT/'verification_large_token_interaction.json', {'passed': True, 'manifest_sha256': digest,
        'cases': cases, 'normal_geometry': diagnostics, 'actual_cuda_full_task_witnesses': witnesses,
        'code_sha256': {name: sha(ROOT/name) for name in ['large_token_interaction.py', 'audit_large_token_interaction.py', 'verify_large_token_interaction.py']},
        'checks': ['Four actual256image fulltasks/allgradients independently restored with nonzero readout',
            'Identical originalnormalL task/init/RNG/basegradients atzero; equalparameters within eachrank',
            'Independent streamnormalPCA, attention, tanh andproduct/addition; tokenpermutation andquery dependence',
            'Bothstream keyupdates/frozenbuffers/strictrestore and fouractualCUDA firsttask/allgradient witnesses'],
        'classification_fits': 0, 'classification_scores': 0, 'external_test_evaluated': False})
    print('Four bounded token interaction full-task cases and four CUDA witnesses passed.', flush=True)


if __name__ == '__main__': main()
