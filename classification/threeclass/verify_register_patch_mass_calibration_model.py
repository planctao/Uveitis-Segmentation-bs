"""Zero-score full-bank, actual256image task and nonlinear-content preflight."""
import json
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
from experiment import ROOT, loss_fn, dump
from kernel_residual_experiments import sha
from register_patch_mass_calibration_data import load_all, fit_mass_statistics, MASS_OUT, MASS_KEY
from register_patch_query_features import array_sha
from register_patch_mass_calibration import RegisterPatchMassCalibrationModel
from register_patch_mass_calibration_witness import tensors_for, weights_for, write_witness, audit_witness, DETAILS
from register_patch_mass_calibration_experiments import restore
from audit_register_patch_mass_calibration import independent_probability
from audit_normal_tangent import complete_task
from register_patch_gated_value import RegisterPatchGatedValueModel


def main():
    torch.set_num_threads(4)
    started = time.time()
    frame, features, bank, tail, digest = load_all()
    fit = np.flatnonzero(frame.split.eq('train'))
    ids = fit[np.random.default_rng(20292900).permutation(len(fit))[:256]]
    source_path = ROOT/'runs/large_token_evidence/registers/validation/seed_0/model.pt'
    source_hash = sha(source_path)
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    plan_path = ROOT/'results/register_patch_mass_calibration_preregistration_before_pair_outcome.json'
    planned = json.loads(plan_path.read_text())
    assert planned['classification_fits'] == planned['scores_evaluated'] == 0
    mean = fit_mass_statistics(bank, fit)
    changed_mass = np.array(bank[MASS_KEY])
    nonfit = np.setdiff1d(np.arange(2091), fit)
    changed_mass[nonfit] = .123
    assert np.array_equal(mean, fit_mass_statistics({MASS_KEY: changed_mass}, fit))
    assert not np.array_equal(mean, changed_mass.mean(axis=(0, 2, 3)))
    cases = []
    for arm in planned['configuration_plan']:
        config = {key: arm[key] for key in ['kind', 'rank', 'alpha', 'mass_mode']}
        model = RegisterPatchMassCalibrationModel(source, tail, config, mean, 20292900).cuda().train()
        names = {'value_down', 'value_gate', 'value_up'}
        assert {name for name, p in model.named_parameters() if p.requires_grad} == names
        before = {name: array_sha(value.cpu().numpy()) for name, value in model.state_dict().items() if name not in names}
        base, selected, registers = tensors_for(features, bank, ids, source, 'cuda')
        y, disease, weights = weights_for(frame, fit, ids, source, 'cuda')
        with torch.no_grad():
            zero = model(base, selected, registers)
            direct = model.head({**base, 'side_tokens': ((registers-model.source_mean_side_tokens)/model.source_std_side_tokens).clamp(-10, 10)})
            assert torch.equal(zero['p'], direct['p']) and torch.equal(zero['subtype'], direct['subtype'])
            assert torch.count_nonzero(zero['tail_difference']) == 0
        with tempfile.TemporaryDirectory(prefix='gated-value-task-') as folder:
            witness_path = Path(folder)/'first_task.npz'
            write_witness(witness_path, model, source, features, bank, frame, fit, ids, config, 0)
            witness = audit_witness(witness_path, source, tail, features, bank, frame, fit, config, 0)
        with torch.no_grad():
            model.value_up.copy_(torch.sin(torch.arange(model.value_up.numel(), device='cuda').reshape_as(model.value_up)*.17)*1e-3)
        torch.cuda.synchronize()
        begin = time.time()
        model.zero_grad()
        out = model(base, selected, registers)
        task = loss_fn(out, y, disease, weights, source['cfg'])
        task.backward()
        torch.cuda.synchronize()
        seconds = time.time()-begin
        assert torch.isfinite(task)
        state = {name: p.detach().clone().requires_grad_() for name, p in model.named_parameters() if p.requires_grad}
        raw = {key: torch.from_numpy(value[ids]).cuda() for key, value in features.items()}
        manual = independent_probability(source, {key: value.cuda() for key, value in tail.state_dict().items()}, state, raw, selected, config, model.fit_mass_mean)
        manual_loss = complete_task(manual, y, disease, weights, source['cfg'])
        manual_loss.backward()
        errors = {}
        for name in DETAILS:
            assert torch.allclose(out[name], manual[name], atol=1e-5, rtol=1e-5), (name, float((out[name]-manual[name]).abs().max()))
        assert torch.equal(out['p'].argmax(1), manual['p'].argmax(1))
        assert abs(float(task.detach())-float(manual_loss.detach())) < 2e-5
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                errors[name] = float((parameter.grad-state[name].grad).abs().max())
                assert torch.allclose(parameter.grad, state[name].grad, atol=1e-6, rtol=3e-3), (name, errors[name])
                assert torch.isfinite(parameter.grad).all() and torch.count_nonzero(parameter.grad) > 0
            else:
                assert parameter.grad is None
        with torch.no_grad():
            order = torch.arange(63, -1, -1, device='cuda')
            changed = {**selected, **{key: selected[key][..., order, :] for key in ['patch_keys', 'patch_values']}}
            permuted = model(base, changed, registers)
            assert torch.allclose(out['p'], permuted['p'], atol=1e-5, rtol=1e-5)
            shortcut = model.patch_update(out['baseline_pool'])*out['applied_attention_mass'][..., None]
            noncommutation = float((out['value_pool_difference']-shortcut).abs().max())
            assert noncommutation > 1e-8, 'The nonlinear patch update must differ from an update after pooling.'
            saved_gate = model.value_gate.clone()
            model.value_gate.zero_()
            closed = model(base, selected, registers)
            assert torch.equal(closed['p'], direct['p']) and torch.equal(closed['subtype'], direct['subtype'])
            assert torch.count_nonzero(closed['value_pool_difference']) == 0
            model.value_gate.copy_(saved_gate)
        with torch.no_grad():
            gated = RegisterPatchGatedValueModel(source, tail, config, 20292900).cuda().eval()
            for name in names:
                getattr(gated, name).copy_(getattr(model, name))
            ones = {**selected, MASS_KEY: torch.ones_like(selected[MASS_KEY])}
            zeros = {**selected, MASS_KEY: torch.zeros_like(selected[MASS_KEY])}
            old_mean = model.fit_mass_mean.clone()
            model.fit_mass_mean.fill_(1.)
            unit = model(base, ones, registers)
            original = gated(base, selected, registers)
            assert torch.equal(unit['p'], original['p']) and torch.equal(unit['subtype'], original['subtype'])
            assert torch.equal(unit['value_pool_difference'], original['value_pool_difference'])
            model.fit_mass_mean.zero_()
            no_mass = model(base, zeros, registers)
            assert torch.equal(no_mass['p'], direct['p']) and torch.equal(no_mass['subtype'], direct['subtype'])
            assert torch.count_nonzero(no_mass['value_pool_difference']) == 0
            model.fit_mass_mean.copy_(old_mean)
            alternate_config = {**config, 'mass_mode': 'constant_mass' if config['mass_mode'] == 'source_mass' else 'source_mass'}
            alternate = RegisterPatchMassCalibrationModel(source, tail, alternate_config, mean, 20292900).cuda().eval()
            for name in names:
                getattr(alternate, name).copy_(getattr(model, name))
            varied = torch.linspace(.01, .99, selected[MASS_KEY].numel(), device='cuda', dtype=torch.float64).reshape_as(selected[MASS_KEY])
            synthetic = {**selected, MASS_KEY: varied}
            varied_out = model(base, synthetic, registers)
            alternate_out = alternate(base, synthetic, registers)
            mass_mode_difference = float((varied_out['value_pool_difference']-alternate_out['value_pool_difference']).abs().max())
            assert mass_mode_difference > 1e-8
            assert not torch.equal(varied_out['p'], alternate_out['p'])
            full_weight_errors = []
            if config['mass_mode'] == 'source_mass':
                verification = json.loads((MASS_OUT/'verification.json').read_text())
                for witness_row in verification['independent_cpu_witnesses']:
                    with np.load(witness_row['path']) as w:
                        wid = w['indices']; view = witness_row['view']
                        q = w['query'].astype('float64'); k = w['all_keys'].astype('float64')
                        logits = q@k.swapaxes(-2, -1)/8
                        numerator = np.exp(logits-logits.max(-1, keepdims=True))
                        full = numerator/numerator.sum(-1, keepdims=True)
                        chosen = np.take_along_axis(full, w['positions'][:, :, None, :].astype('int64')+5, -1)
                        values = torch.from_numpy(np.array(bank['patch_values'][wid, view:view+1])).cuda()
                        updates = model.patch_update(values).cpu().numpy()[:, 0]
                        direct_pool = chosen@updates
                        selected_keys = np.array(bank['patch_keys'][wid, view]).astype('float64')
                        conditional_logits = q@selected_keys.swapaxes(-2, -1)/8
                        local_exp = np.exp(conditional_logits-conditional_logits.max(-1, keepdims=True))
                        local = local_exp/local_exp.sum(-1, keepdims=True)
                        weighted_pool = (local@updates)*np.array(bank[MASS_KEY][wid, view])[..., None]
                        error = float(np.abs(direct_pool-weighted_pool).max())
                        assert np.allclose(direct_pool, weighted_pool, atol=1e-12, rtol=1e-11), error
                        full_weight_errors.append(error)
            del gated, alternate, unit, original, no_mass, varied_out, alternate_out
        saved = {'fit_mass_mean': mean.copy(), 'fit_indices': fit.copy(), 'source_checkpoint': str(source_path), 'source_checkpoint_sha256': source_hash,
            'adapter_config': config, 'seed': 0,
            'value_state_dict': {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}}
        with tempfile.TemporaryDirectory(prefix='gated-value-restore-') as folder:
            path = Path(folder)/'model.pt'
            torch.save(saved, path)
            restored, _ = restore(torch.load(path, map_location='cpu', weights_only=False), tail)
            with torch.inference_mode():
                replay = restored(base, selected, registers)
            assert torch.equal(replay['p'], out['p']) and torch.equal(replay['subtype'], out['subtype'])
        torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=.01).step()
        for name, value in model.state_dict().items():
            if name in before:
                assert array_sha(value.cpu().numpy()) == before[name]
        cases.append({'configuration': config, 'actual_images': len(ids),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'first_task_witness': witness, 'nonzero_adapter_complete_task_gradient_errors': errors,
            'maximum_probability_error': float((out['p']-manual['p']).abs().max()),
            'full_batch_forward_backward_seconds': seconds, 'exact_zero_source_probability_and_subtype': True,
            'independent_headwise_perpatch_formula_verified': True, 'noncommutation_with_attention_pooling_max_difference': noncommutation,
            'mass_one_exactly_matches_preserved_gated_model': True,
            'mass_zero_exact_source_probability_and_subtype': True,
            'source_constant_synthetic_pool_difference': mass_mode_difference,
            'six_saved_full_source_weight_algebra_errors': full_weight_errors,
            'joint_patch_permutation_invariance': True, 'closed_gate_exact_source_probability_and_subtype': True,
            'checkpoint_restore_bitwise_identical': True, 'all_frozen_head_tail_buffers_unchanged_after_optimizer': True})
        print(json.dumps({'event': 'gated_value_preflight', 'config': config, 'gradient_errors': errors,
            'noncommutation': noncommutation, 'seconds': seconds}), flush=True)
        del model, restored, out, manual, task, manual_loss, zero, direct, permuted, replay, closed, shortcut, saved_gate, state
        torch.cuda.empty_cache()
    assert sha(source_path) == source_hash
    code = ['register_patch_mass_calibration.py', 'audit_register_patch_mass_calibration.py', 'register_patch_mass_calibration_witness.py',
        'register_patch_mass_calibration_experiments.py', 'register_patch_mass_calibration_data.py', 'verify_register_patch_mass_calibration_model.py', 'patch_adapter_cuda_ready.py']
    dump(ROOT/'verification_register_patch_mass_calibration_model.json', {'passed': True, 'manifest_sha256': digest,
        'cases': cases, 'fit_only_indices': ids.tolist(), 'source_checkpoint_sha256': {str(source_path): source_hash},
        'preregistration_sha256': sha(plan_path), 'bank_verification_sha256': sha(ROOT/'verification_register_patch_query_bank.json'),
        'mass_bank_verification_sha256': sha(MASS_OUT/'verification.json'),
        'mass_bank_root_review_sha256': sha(ROOT/'results/register_patch_attention_mass_bank_v2_root_review.json'),
        'fit_only_mass_statistics_nonfit_perturbation_invariant': True, 'fit_mass_mean': mean.tolist(),
        'classification_fits': 0, 'classification_scores': 0, 'external_images_read': 0,
        'code_sha256': {name: sha(ROOT/name) for name in code}, 'seconds': time.time()-started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
    print('Four actual256image nonlinear patch content cases passed; zero classifier fits/scores.', flush=True)


if __name__ == '__main__':
    main()
