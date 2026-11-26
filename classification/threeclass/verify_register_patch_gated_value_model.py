"""Zero-score full-bank, actual256image task and nonlinear-content preflight."""
import json
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
from experiment import ROOT, loss_fn, dump
from kernel_residual_experiments import sha
from register_patch_query_data import load_all
from register_patch_query_features import array_sha
from register_patch_gated_value import RegisterPatchGatedValueModel
from register_patch_gated_value_witness import tensors_for, weights_for, write_witness, audit_witness, DETAILS
from register_patch_gated_value_experiments import restore
from audit_register_patch_gated_value import independent_probability
from audit_normal_tangent import complete_task


def main():
    torch.set_num_threads(4)
    started = time.time()
    frame, features, bank, tail, digest = load_all()
    fit = np.flatnonzero(frame.split.eq('train'))
    ids = fit[np.random.default_rng(20292300).permutation(len(fit))[:256]]
    source_path = ROOT/'runs/large_token_evidence/registers/validation/seed_0/model.pt'
    source_hash = sha(source_path)
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    plan_path = ROOT/'results/register_patch_gated_value_preregistration_before_value_joint_outcomes.json'
    planned = json.loads(plan_path.read_text())
    assert planned['classification_fits'] == planned['scores_evaluated'] == 0
    cases = []
    for arm in planned['configuration_plan']:
        config = {key: arm[key] for key in ['kind', 'rank', 'alpha']}
        model = RegisterPatchGatedValueModel(source, tail, config, 20292300).cuda().train()
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
        manual = independent_probability(source, {key: value.cuda() for key, value in tail.state_dict().items()}, state, raw, selected, config)
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
            shortcut = model.patch_update(out['baseline_pool'])
            noncommutation = float((out['value_pool_difference']-shortcut).abs().max())
            assert noncommutation > 1e-8, 'The nonlinear patch update must differ from an update after pooling.'
            saved_gate = model.value_gate.clone()
            model.value_gate.zero_()
            closed = model(base, selected, registers)
            assert torch.equal(closed['p'], direct['p']) and torch.equal(closed['subtype'], direct['subtype'])
            assert torch.count_nonzero(closed['value_pool_difference']) == 0
            model.value_gate.copy_(saved_gate)
        saved = {'source_checkpoint': str(source_path), 'source_checkpoint_sha256': source_hash,
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
            'joint_patch_permutation_invariance': True, 'closed_gate_exact_source_probability_and_subtype': True,
            'checkpoint_restore_bitwise_identical': True, 'all_frozen_head_tail_buffers_unchanged_after_optimizer': True})
        print(json.dumps({'event': 'gated_value_preflight', 'config': config, 'gradient_errors': errors,
            'noncommutation': noncommutation, 'seconds': seconds}), flush=True)
        del model, restored, out, manual, task, manual_loss, zero, direct, permuted, replay, closed, shortcut, saved_gate, state
        torch.cuda.empty_cache()
    assert sha(source_path) == source_hash
    code = ['register_patch_gated_value.py', 'audit_register_patch_gated_value.py', 'register_patch_gated_value_witness.py',
        'register_patch_gated_value_experiments.py', 'verify_register_patch_gated_value_model.py', 'patch_adapter_cuda_ready.py']
    dump(ROOT/'verification_register_patch_gated_value_model.json', {'passed': True, 'manifest_sha256': digest,
        'cases': cases, 'fit_only_indices': ids.tolist(), 'source_checkpoint_sha256': {str(source_path): source_hash},
        'preregistration_sha256': sha(plan_path), 'bank_verification_sha256': sha(ROOT/'verification_register_patch_query_bank.json'),
        'classification_fits': 0, 'classification_scores': 0, 'external_images_read': 0,
        'code_sha256': {name: sha(ROOT/name) for name in code}, 'seconds': time.time()-started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
    print('Four actual256image nonlinear patch content cases passed; zero classifier fits/scores.', flush=True)


if __name__ == '__main__':
    main()
