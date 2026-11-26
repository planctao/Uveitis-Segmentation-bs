"""Actual random256-image joint query/value reductions, full tasks and gradients."""
import json
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
from experiment import ROOT, loss_fn, dump
from kernel_residual_experiments import sha
from register_patch_query_data import load_all
from register_patch_query import RegisterPatchQueryModel
from register_patch_value import RegisterPatchValueModel
from register_patch_joint import RegisterPatchJointModel
from register_patch_joint_witness import tensors_for, weights_for, write_witness, audit_witness
from register_patch_joint_experiments import restore
from register_patch_query_features import array_sha
from audit_register_patch_joint import independent_probability
from audit_normal_tangent import complete_task


def initialize_nonzero(model):
    with torch.no_grad():
        for name in ['up.weight', 'value_up']:
            parameter = dict(model.named_parameters())[name]
            parameter.copy_(torch.sin(torch.arange(parameter.numel(), device=parameter.device).reshape_as(parameter)*.17)*1e-3)


def main():
    torch.set_num_threads(4)
    started = time.time()
    frame, features, bank, tail, digest = load_all()
    fit = np.flatnonzero(frame.split.eq('train'))
    ids = fit[np.random.default_rng(20292100).permutation(len(fit))[:256]]
    source_path = ROOT/'runs/large_token_evidence/registers/validation/seed_0/model.pt'
    source_hash = sha(source_path)
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    hypothesis_path = ROOT/'results/next_register_patch_joint_hypothesis.json'
    hypothesis = json.loads(hypothesis_path.read_text())
    assert hypothesis['classification_fits'] == hypothesis['scores_evaluated'] == 0
    cases = []
    for arm in hypothesis['configuration_plan']:
        config = {key: arm[key] for key in ['query_kind', 'value_kind', 'rank', 'alpha']}
        model = RegisterPatchJointModel(source, tail, config, 20292100).cuda().train()
        names = {'down.weight', 'up.weight', 'value_down', 'value_up'}
        assert {name for name, p in model.named_parameters() if p.requires_grad} == names
        frozen = {name: array_sha(value.detach().cpu().numpy()) for name, value in model.state_dict().items() if name not in names}
        base, selected, registers = tensors_for(features, bank, ids, source, 'cuda')
        y, disease, weights = weights_for(frame, fit, ids, source, 'cuda')
        with tempfile.TemporaryDirectory(prefix='patch-joint-witness-') as folder:
            path = Path(folder)/'first_task.npz'
            write_witness(path, model, source, features, bank, frame, fit, ids, config, 0)
            witness = audit_witness(path, source, tail, features, bank, frame, fit, config, 0)
        with torch.no_grad():
            zero = model(base, selected, registers)
            direct = model.head({**base, 'side_tokens': ((registers-model.source_mean_side_tokens)/model.source_std_side_tokens).clamp(-10, 10)})
            assert torch.equal(zero['p'], direct['p']) and torch.equal(zero['subtype'], direct['subtype'])
            assert torch.count_nonzero(zero['tail_difference']) == 0
            initialize_nonzero(model)
            query_config = {'kind': config['query_kind'], 'rank': config['rank'], 'alpha': config['alpha']}
            query = RegisterPatchQueryModel(source, tail, query_config, 20292100).cuda().eval()
            query.down.weight.copy_(model.down.weight)
            query.up.weight.copy_(model.up.weight)
            value_up = model.value_up.clone()
            model.value_up.zero_()
            single = query(base, selected, registers)
            reduced = model(base, selected, registers)
            assert torch.equal(single['p'], reduced['p']) and torch.equal(single['tail_difference'], reduced['tail_difference'])
            model.value_up.copy_(value_up)
            query_up = model.up.weight.clone()
            model.up.weight.zero_()
            value_config = {'kind': config['value_kind'], 'rank': config['rank'], 'alpha': config['alpha']}
            value = RegisterPatchValueModel(source, tail, value_config, 20292100).cuda().eval()
            value.value_down.copy_(model.value_down)
            value.value_up.copy_(model.value_up)
            single = value(base, selected, registers)
            reduced = model(base, selected, registers)
            assert torch.equal(single['p'], reduced['p']) and torch.equal(single['tail_difference'], reduced['tail_difference'])
            model.up.weight.copy_(query_up)
        del zero, direct, query, value, single, reduced
        model.zero_grad()
        torch.cuda.synchronize()
        begin = time.time()
        out = model(base, selected, registers)
        task = loss_fn(out, y, disease, weights, source['cfg'])
        assert torch.isfinite(task)
        task.backward()
        torch.cuda.synchronize()
        seconds = time.time()-begin
        independent_state = {name: p.detach().clone().requires_grad_() for name, p in model.named_parameters() if p.requires_grad}
        raw = {key: torch.from_numpy(features[key][ids]).cuda() for key in features}
        tail_state = {name: value.cuda() for name, value in tail.state_dict().items()}
        manual = independent_probability(source, tail_state, independent_state, raw, selected, config)
        manual_task = complete_task(manual, y, disease, weights, source['cfg'])
        manual_task.backward()
        trace_errors = {}
        for name in ['delta_query', 'selected_attention', 'baseline_attention', 'query_pool_difference', 'value_pool_difference', 'pool_difference', 'attention_difference', 'tail_difference', 'p', 'subtype']:
            trace_errors[name] = float((out[name]-manual[name]).abs().max())
            assert torch.allclose(out[name], manual[name], atol=1e-5, rtol=1e-5), (name, trace_errors[name])
        assert torch.equal(out['p'].argmax(1), manual['p'].argmax(1))
        assert abs(float(task.detach())-float(manual_task.detach())) < 2e-5
        gradients = {}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                gradients[name] = float((parameter.grad-independent_state[name].grad).abs().max())
                assert torch.allclose(parameter.grad, independent_state[name].grad, atol=1e-6, rtol=3e-3), (name, gradients[name])
                assert torch.isfinite(parameter.grad).all() and torch.count_nonzero(parameter.grad) > 0
            else:
                assert parameter.grad is None
        checkpoint = {'source_checkpoint': str(source_path), 'source_checkpoint_sha256': source_hash,
            'adapter_config': config, 'seed': 0,
            'joint_state_dict': {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}}
        restored, _ = restore(checkpoint, tail)
        restored = restored.cuda().eval()
        with torch.no_grad():
            assert torch.equal(restored(base, selected, registers)['p'], out['p'])
            permutation = torch.arange(63, -1, -1, device='cuda')
            permuted = dict(selected)
            for name in ['patch_keys', 'patch_values']:
                permuted[name] = selected[name][..., permutation, :]
            permuted_out = model(base, permuted, registers)
            assert torch.allclose(permuted_out['p'], out['p'], atol=1e-5, rtol=1e-5)
            # Freeze attention at the original queries to isolate the Q/V interaction.
            interaction_free = out['baseline_attention']@selected['patch_values'].double()
            code = torch.einsum('bshqd,hrd->bshqr', interaction_free, model.value_down.double().expand(16, -1, -1))
            no_interaction = torch.einsum('bshqr,hdr->bshqd', code, model.value_up.double().expand(16, -1, -1))*(config['alpha']/config['rank'])
            interaction_size = float((out['value_pool_difference']-no_interaction).abs().max())
            assert interaction_size > 0
        torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=.01).step()
        for name, value in model.state_dict().items():
            if name in frozen:
                assert array_sha(value.detach().cpu().numpy()) == frozen[name]
        cases.append({'configuration': config, 'actual_images': len(ids), 'first_task_witness': witness,
            'complete_nonzero_task_gradient_errors': gradients, 'trace_errors': trace_errors,
            'query_value_interaction_magnitude': interaction_size, 'full_batch_forward_backward_seconds': seconds,
            'exact_zero_source_identity': True, 'exact_query_only_and_value_only_reductions': True,
            'joint_patch_permutation_invariance': True, 'checkpoint_restore_bitwise_identical': True,
            'frozen_head_tail_buffers_unchanged_after_optimizer': True})
        print(json.dumps({'event': 'joint_patch_preflight', 'configuration': config,
            'maximum_gradient_error': max(gradients.values()), 'seconds': seconds}), flush=True)
        del model, out, manual, task, manual_task, restored, permuted_out
        torch.cuda.empty_cache()
    assert sha(source_path) == source_hash
    code = ['register_patch_joint.py', 'audit_register_patch_joint.py', 'register_patch_joint_witness.py',
        'register_patch_joint_experiments.py', 'verify_register_patch_joint_model.py']
    dump(ROOT/'verification_register_patch_joint_model.json', {'passed': True, 'manifest_sha256': digest,
        'cases': cases, 'classification_fits': 0, 'classification_scores': 0, 'external_images_read': 0,
        'bank_verification_sha256': sha(ROOT/'verification_register_patch_query_bank.json'),
        'preregistered_hypothesis_sha256': sha(hypothesis_path),
        'source_checkpoint_sha256': {str(source_path): source_hash}, 'fit_only_indices': ids.tolist(),
        'code_sha256': {name: sha(ROOT/name) for name in code}, 'seconds': time.time()-started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
    print('Four random256-image joint task, ablation, interaction and gradient preflights passed; zero fits/scores.', flush=True)


if __name__ == '__main__': main()
