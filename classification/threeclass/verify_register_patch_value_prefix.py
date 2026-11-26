"""Zero-score actual256-image feasibility for queued patch-value adaptation."""
import json
import time
import numpy as np
import torch
from experiment import ROOT, loss_fn, dump
from kernel_residual_experiments import sha
from register_multiquery_data import load_all
from register_patch_query_data import load_tail
from register_patch_query_features import BANK_KEYS, array_sha
from register_patch_query_witness import tensors_for, weights_for
from register_patch_value import RegisterPatchValueModel
from audit_register_patch_value import independent_probability
from audit_normal_tangent import complete_task


def main():
    torch.set_num_threads(4)
    started = time.time()
    frame, features, digest = load_all()
    fit = np.flatnonzero(frame.split.eq('train'))
    ids = fit[:256]
    bank_folder = ROOT/'runs/register_patch_query_bank'
    bank = {key: np.load(bank_folder/'arrays'/(key+'.npy'), mmap_mode='r') for key in BANK_KEYS}
    chunks = {}
    for begin in sorted(set((ids//8)*8)):
        path = bank_folder/'chunks'/f'{begin:05d}.json'
        proof = json.loads(path.read_text())
        for key in BANK_KEYS:
            assert array_sha(np.asarray(bank[key][proof['indices']])) == proof['array_sha256'][key]
        chunks[str(path)] = sha(path)
    tail = load_tail(bank_folder/'frozen_tail.pt')
    source_path = ROOT/'runs/large_token_evidence/registers/validation/seed_0/model.pt'
    source_hash = sha(source_path)
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    plan_path = ROOT/'results/next_register_patch_value_hypothesis.json'
    planned = json.loads(plan_path.read_text())
    assert planned['classification_fits'] == planned['scores_evaluated'] == 0
    cases = []
    for arm in planned['configuration_plan']:
        config = {key: arm[key] for key in ['kind', 'rank', 'alpha']}
        model = RegisterPatchValueModel(source, tail, config, 20291900).cuda().train()
        expected_names = {'value_down', 'value_up'}
        assert {name for name, p in model.named_parameters() if p.requires_grad} == expected_names
        before = {name: array_sha(value.detach().cpu().numpy()) for name, value in model.state_dict().items() if name not in expected_names}
        base, selected, registers = tensors_for(features, bank, ids, source, 'cuda')
        y, disease, weights = weights_for(frame, fit, ids, source, 'cuda')
        with torch.no_grad():
            zero = model(base, selected, registers)
            direct = model.head({**base, 'side_tokens': ((registers-model.source_mean_side_tokens)/model.source_std_side_tokens).clamp(-10, 10)})
            assert torch.equal(zero['p'], direct['p']) and torch.equal(zero['subtype'], direct['subtype'])
            assert torch.count_nonzero(zero['tail_difference']) == 0
            model.value_up.copy_(torch.sin(torch.arange(model.value_up.numel(), device='cuda').reshape_as(model.value_up)*.17)*1e-3)
        torch.cuda.synchronize()
        begin = time.time()
        out = model(base, selected, registers)
        task = loss_fn(out, y, disease, weights, source['cfg'])
        task.backward()
        torch.cuda.synchronize()
        seconds = time.time()-begin
        assert torch.isfinite(task)
        state = {name: p.detach().clone().requires_grad_() for name, p in model.named_parameters() if p.requires_grad}
        raw = {key: torch.from_numpy(value[ids]).cuda() for key, value in features.items()}
        manual = independent_probability(source, {key: value.cuda() for key, value in tail.state_dict().items()}, state, raw, selected, config)
        independent_loss = complete_task(manual, y, disease, weights, source['cfg'])
        independent_loss.backward()
        errors = {}
        for name in ['selected_attention', 'baseline_pool', 'value_pool_difference', 'pool_difference', 'attention_difference', 'tail_difference', 'p', 'subtype']:
            assert torch.allclose(out[name], manual[name], atol=1e-5, rtol=1e-5), (name, float((out[name]-manual[name]).abs().max()))
        assert torch.equal(out['p'].argmax(1), manual['p'].argmax(1))
        assert abs(float(task.detach())-float(independent_loss.detach())) < 2e-5
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                errors[name] = float((parameter.grad-state[name].grad).abs().max())
                assert torch.allclose(parameter.grad, state[name].grad, atol=1e-6, rtol=3e-3), (name, errors[name])
                assert torch.isfinite(parameter.grad).all() and torch.count_nonzero(parameter.grad) > 0
            else:
                assert parameter.grad is None
        with torch.no_grad():
            # The same value map is shared across scales and registers, so register/patch order is immaterial.
            order = torch.arange(63, -1, -1, device='cuda')
            changed = {key: value for key, value in selected.items()}
            for key in ['patch_keys', 'patch_values']:
                changed[key] = selected[key][..., order, :]
            permuted = model(base, changed, registers)
            assert torch.allclose(out['p'], permuted['p'], atol=1e-5, rtol=1e-5)
            if config['kind'] == 'shared':
                assert model.value_down.shape[0] == model.value_up.shape[0] == 1
        torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=.01).step()
        for name, value in model.state_dict().items():
            if name in before:
                assert array_sha(value.detach().cpu().numpy()) == before[name]
        cases.append({'configuration': config, 'actual_images': len(ids),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'complete_task_gradient_errors': errors, 'maximum_probability_error': float((out['p']-manual['p']).abs().max()),
            'full_batch_forward_backward_seconds': seconds, 'exact_zero_source_probability_and_subtype': True,
            'independent_perpatch_vs_production_pooled_value_update': True,
            'joint_patch_permutation_invariance': True, 'all_frozen_head_tail_and_buffers_unchanged_after_optimizer': True})
        print(json.dumps({'event': 'patch_value_prefix256', 'config': config, 'gradient_errors': errors, 'seconds': seconds}), flush=True)
        del model, out, manual, task, independent_loss, zero, direct, permuted
        torch.cuda.empty_cache()
    assert sha(source_path) == source_hash
    code = ['register_patch_value.py', 'audit_register_patch_value.py', 'register_patch_value_witness.py',
        'register_patch_value_experiments.py', 'verify_register_patch_value_prefix.py']
    dump(ROOT/'verification_register_patch_value_prefix256.json', {'passed': True, 'manifest_sha256': digest,
        'cases': cases, 'fit_only_indices': ids.tolist(), 'chunk_proof_sha256': chunks,
        'source_checkpoint_sha256': {str(source_path): source_hash}, 'preregistered_hypothesis_sha256': sha(plan_path),
        'classification_fits': 0, 'classification_scores': 0, 'external_images_read': 0,
        'full_bank_random_firsttask_preflight_and_predecessor_terminal_required': True,
        'code_sha256': {name: sha(ROOT/name) for name in code}, 'seconds': time.time()-started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
    print('Four actual256-image patch-value feasibility cases passed; zero classifier fits/scores.', flush=True)


if __name__ == '__main__': main()
