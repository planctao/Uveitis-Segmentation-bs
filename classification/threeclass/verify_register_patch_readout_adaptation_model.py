"""Zero-score head-only control and local-content/readout complete-task preflight."""
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
from register_patch_readout_adaptation import RegisterPatchReadoutAdaptationModel
from register_patch_readout_adaptation_witness import tensors_for, weights_for, write_witness, audit_witness, DETAILS
from register_patch_readout_adaptation_experiments import restore
from audit_register_patch_readout_adaptation import independent_probability
from audit_normal_tangent import complete_task


def main():
    torch.set_num_threads(4)
    started = time.time()
    frame, features, bank, tail, digest = load_all()
    fit = np.flatnonzero(frame.split.eq('train'))
    ids = fit[np.random.default_rng(20292500).permutation(len(fit))[:256]]
    source_path = ROOT/'runs/large_token_evidence/registers/validation/seed_0/model.pt'
    source_hash = sha(source_path)
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    plan_path = ROOT/'results/register_patch_readout_adaptation_preregistration_before_value_joint_gated_outcomes.json'
    planned = json.loads(plan_path.read_text())
    assert planned['classification_fits'] == planned['scores_evaluated'] == 0
    cases = []
    for arm in planned['configuration_plan']:
        config = {key: arm[key] for key in ['kind', 'rank', 'alpha', 'adapt_patches']}
        model = RegisterPatchReadoutAdaptationModel(source, tail, config, 20292500).cuda().train()
        names = {'head.flat.delta.weight','head.flat.delta.bias'}
        if config['adapt_patches']: names.update(['value_down','value_gate','value_up'])
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
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and (name=='value_up' or name.startswith('head.flat.delta.')):
                    parameter.copy_(torch.sin(torch.arange(parameter.numel(), device='cuda').reshape_as(parameter)*.17)*1e-3)
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
            # With the patch update closed, the exact same learned readout acts on original features.
            saved_up = model.value_up.clone()
            model.value_up.zero_()
            closed = model(base, selected, registers)
            readout_only = model.head({**base, 'side_tokens': ((registers-model.source_mean_side_tokens)/model.source_std_side_tokens).clamp(-10, 10)})
            assert torch.equal(closed['p'], readout_only['p']) and torch.equal(closed['subtype'], readout_only['subtype'])
            assert torch.count_nonzero(closed['value_pool_difference']) == 0
            model.value_up.copy_(saved_up)
            # With the readout delta closed, recover the original gated-value engine exactly.
            from register_patch_gated_value import RegisterPatchGatedValueModel
            gated = RegisterPatchGatedValueModel(source, tail, config, 20292500).cuda().eval()
            for name in ['value_down','value_gate','value_up']:
                getattr(gated,name).copy_(getattr(model,name))
            gated_output = gated(base, selected, registers)
            saved_delta = {name:parameter.clone() for name,parameter in model.head.flat.delta.named_parameters()}
            for parameter in model.head.flat.delta.parameters(): parameter.zero_()
            closed_readout = model(base, selected, registers)
            assert torch.equal(closed_readout['p'],gated_output['p']) and torch.equal(closed_readout['subtype'],gated_output['subtype'])
            for name,parameter in model.head.flat.delta.named_parameters(): parameter.copy_(saved_delta[name])
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
            'independent_headwise_patch_formula_and_functional_readout_delta_verified': True,
            'joint_patch_permutation_invariance': True, 'closed_patch_path_exact_head_only_reduction': True, 'closed_readout_exact_gated_value_reduction': True,
            'checkpoint_restore_bitwise_identical': True, 'all_frozen_head_tail_buffers_unchanged_after_optimizer': True})
        print(json.dumps({'event': 'gated_value_preflight', 'config': config, 'gradient_errors': errors,
            'trainable_tensors':len(names), 'seconds': seconds}), flush=True)
        del model, restored, out, manual, task, manual_loss, zero, direct, permuted, replay, closed, readout_only, saved_up, gated, gated_output, closed_readout, saved_delta, state
        torch.cuda.empty_cache()
    assert sha(source_path) == source_hash
    code = ['register_patch_readout_adaptation.py', 'audit_register_patch_readout_adaptation.py', 'register_patch_readout_adaptation_witness.py',
        'register_patch_readout_adaptation_experiments.py', 'verify_register_patch_readout_adaptation_model.py', 'patch_adapter_cuda_ready.py',
        'register_patch_gated_value.py', 'audit_register_patch_gated_value.py']
    dump(ROOT/'verification_register_patch_readout_adaptation_model.json', {'passed': True, 'manifest_sha256': digest,
        'cases': cases, 'fit_only_indices': ids.tolist(), 'source_checkpoint_sha256': {str(source_path): source_hash},
        'preregistration_sha256': sha(plan_path), 'bank_verification_sha256': sha(ROOT/'verification_register_patch_query_bank.json'),
        'classification_fits': 0, 'classification_scores': 0, 'external_images_read': 0,
        'code_sha256': {name: sha(ROOT/name) for name in code}, 'seconds': time.time()-started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
    print('Four actual256image readout/control and joint patch content cases passed; zero classifier fits/scores.', flush=True)


if __name__ == '__main__':
    main()
