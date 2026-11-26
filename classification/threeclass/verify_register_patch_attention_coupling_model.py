"""Four actual256-image joint-reader engineering tasks; no candidate fitting/scoring."""
import json
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
from experiment import ROOT, loss_fn, dump
from kernel_residual_experiments import sha
from audit_normal_tangent import complete_task
from register_patch_query_features import array_sha
from register_patch_query_selection_data import load_all
from register_patch_query_selection import RegisterPatchQuerySelectionModel
from register_patch_attention_coupling import RegisterPatchAttentionCouplingModel, reconstruction_state, restore, PATCH_NAMES, ATTENTION_PREFIX
from register_patch_attention_coupling_witness import write_witness, audit_witness, tensors_for, weights_for
from audit_register_patch_attention_coupling import independent_probability, independent_reader

OUT = ROOT/'runs/register_patch_attention_coupling_preflight'


def verify(protocol):
    for section in ['code_sha256', 'input_artifact_sha256']:
        for name, expected in protocol[section].items():
            path = Path(name) if Path(name).is_absolute() else ROOT/name
            assert sha(path) == expected, name


def main():
    destination = ROOT/'verification_register_patch_attention_coupling_model.json'
    assert not destination.exists()
    torch.set_num_threads(4)
    started = time.time()
    protocol = json.loads((OUT/'protocol.json').read_text())
    verify(protocol)
    prereg_path = ROOT/'results/register_patch_attention_coupling_preregistration_before_query_outcome.json'
    planned = json.loads(prereg_path.read_text())
    assert planned['classification_fits'] == planned['scores_evaluated'] == 0
    frame, features, bank, tail, digest = load_all()
    fit = np.flatnonzero(frame.split.eq('train'))
    ids = fit[np.random.default_rng(20292900).permutation(len(fit))[:256]]
    assert frame.iloc[ids].split.eq('train').all()
    source_path = ROOT/'runs/large_token_evidence/registers/validation/seed_0/model.pt'
    source_hash = sha(source_path)
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    base, selected, registers = tensors_for(features, bank, ids, source, 'cuda')
    raw = {key: torch.from_numpy(value[ids]).cuda() for key, value in features.items()}
    y, disease, weights = weights_for(frame, fit, ids, source, 'cuda')
    cases = []
    for arm in planned['configuration_plan']:
        config = {key: arm[key] for key in ['kind', 'rank', 'alpha', 'selection', 'adapt_attention', 'adapt_patches']}
        model = RegisterPatchAttentionCouplingModel(source, tail, config, 20292900).cuda().train()
        names = {name for name, p in model.named_parameters() if p.requires_grad}
        expected = {ATTENTION_PREFIX+name for name, _ in model.head.token_evidence.named_parameters() if name != 'key.bias'}
        if config['adapt_patches']:
            expected.update(PATCH_NAMES)
        assert names == expected and len(names) == (14 if config['adapt_patches'] else 11)
        frozen = {name: array_sha(value.cpu().numpy()) for name, value in model.state_dict().items() if name not in names}
        parent = RegisterPatchQuerySelectionModel(source, tail, config, 20292900).cuda().eval()
        with torch.no_grad():
            initial = model(base, selected, registers)
            original = parent(base, selected, registers)
            assert torch.equal(initial['p'], original['p']) and torch.equal(initial['subtype'], original['subtype'])
            assert torch.count_nonzero(initial['tail_difference']) == 0
        path = OUT/'cases'/(arm['name']+'_actual_first_task.npz')
        path.parent.mkdir(exist_ok=True)
        assert not path.exists()
        write_witness(path, model, source, features, bank, frame, fit, ids, config, 0)
        first = audit_witness(path, source, tail, features, bank, frame, fit, config, 0)
        with torch.no_grad():
            if config['adapt_patches']:
                model.value_up.copy_(torch.sin(torch.arange(model.value_up.numel(), device='cuda').reshape_as(model.value_up)*.17)*1e-3)
            reader = model.head.token_evidence
            for parameter in [reader.query.weight, reader.key.weight, reader.value.weight, reader.readout.weight]:
                parameter.add_(torch.cos(torch.arange(parameter.numel(), device='cuda').reshape_as(parameter)*.13)*1e-4)
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        begin = time.time()
        output = model(base, selected, registers)
        task = loss_fn(output, y, disease, weights, source['cfg'])
        assert torch.isfinite(task)
        task.backward()
        torch.cuda.synchronize()
        seconds = time.time()-begin
        state = reconstruction_state(model, differentiable=True)
        manual = independent_probability(source, model.tail.state_dict(), state, raw, selected, config)
        manual_loss = complete_task(manual, y, disease, weights, source['cfg'])
        manual_loss.backward()
        assert torch.allclose(output['p'], manual['p'], atol=5e-5, rtol=5e-5)
        assert torch.equal(output['p'].argmax(1), manual['p'].argmax(1))
        assert abs(float(task.detach())-float(manual_loss.detach())) < 2e-5
        errors = {}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                error = float((parameter.grad-state[name].grad).abs().max())
                assert torch.allclose(parameter.grad, state[name].grad, atol=1e-6, rtol=3e-3), (name, error)
                assert torch.isfinite(parameter.grad).all() and torch.count_nonzero(parameter.grad) > 0, name
                errors[name] = error
            else:
                assert parameter.grad is None
        with torch.no_grad():
            reader = independent_reader(state, output['token_projected'], output['token_base_hidden'])
            reader_errors = {}
            for name, value in reader.items():
                error = float((value-output[name]).abs().max())
                assert torch.allclose(value, output[name], atol=5e-5, rtol=5e-5), (name, error)
                reader_errors[name] = error
            prefix = 'query16_' if config['selection'] == 'query_top16' else 'shared16_'
            order = torch.arange(15, -1, -1, device='cuda')
            permuted = model(base, {**selected, **{prefix+key: selected[prefix+key][..., order, :] for key in ['keys', 'values']}}, registers)
            assert torch.allclose(output['p'], permuted['p'], atol=1e-5, rtol=1e-5)
            frozen_reader_config = {**config, 'adapt_attention': False, 'adapt_patches': True}
            frozen_reader = RegisterPatchAttentionCouplingModel(source, tail, frozen_reader_config, 20292900).cuda().eval()
            for name in PATCH_NAMES:
                getattr(frozen_reader, name).copy_(getattr(model, name))
                getattr(parent, name).copy_(getattr(model, name))
            as_parent = parent(base, selected, registers)
            as_frozen_reader = frozen_reader(base, selected, registers)
            assert torch.equal(as_parent['p'], as_frozen_reader['p'])
            assert torch.equal(as_parent['subtype'], as_frozen_reader['subtype'])
            attention_effect = float((output['p']-as_frozen_reader['p']).abs().max())
            assert attention_effect > 0
            if not config['adapt_patches']:
                assert torch.count_nonzero(output['tail_difference']) == 0 and torch.count_nonzero(model.value_up) == 0
        checkpoint = {'adapter_config': config, 'seed': 0, 'fit_indices': fit.copy(),
            'source_checkpoint': str(source_path), 'source_checkpoint_sha256': source_hash,
            'trainable_state_dict': {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}}
        with tempfile.TemporaryDirectory(prefix='attention-coupling-restore-') as directory:
            save = Path(directory)/'model.pt'
            torch.save(checkpoint, save)
            restored, _ = restore(torch.load(save, map_location='cpu', weights_only=False), tail)
            with torch.inference_mode():
                replay = restored(base, selected, registers)
            assert torch.equal(replay['p'], output['p']) and torch.equal(replay['subtype'], output['subtype'])
        groups = model.parameter_groups()
        assert [group['lr'] for group in groups] == ([.0001, .001] if config['adapt_patches'] else [.0001])
        torch.optim.AdamW(groups, weight_decay=.01).step()
        for name, value in model.state_dict().items():
            if name in frozen:
                assert array_sha(value.cpu().numpy()) == frozen[name], name
        if not config['adapt_patches']:
            assert torch.count_nonzero(model.value_up) == 0
        cases.append({'configuration': config, 'actual_images': len(ids),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'first_task_witness': first, 'first_task_witness_path': str(path), 'first_task_witness_sha256': sha(path),
            'nonzero_complete_task_gradient_errors': errors, 'independent_reader_trace_errors': reader_errors,
            'maximum_probability_error': float((output['p']-manual['p']).abs().max()),
            'actual_forward_backward_seconds': seconds, 'source_probability_and_subtype_bitwise_at_initialization': True,
            'attention_frozen_reduction_to_parent_bitwise': True, 'nonzero_attention_probability_effect': attention_effect,
            'joint_patch_permutation_verified': True, 'ordinary_cuda_checkpoint_restore_bitwise': True,
            'all_frozen_base_geometry_tail_and_key_bias_buffers_unchanged': True,
            'head_only_patch_zero_preserved': not config['adapt_patches'], 'optimizer_learning_rates': [g['lr'] for g in groups]})
        print(json.dumps({'arm': arm['name'], 'actual_images': len(ids), 'maximum_gradient_error': max(errors.values())}), flush=True)
        del model, parent, initial, original, output, task, state, manual, manual_loss, reader, permuted, frozen_reader, as_parent, as_frozen_reader, restored, replay
        torch.cuda.empty_cache()
    verify(protocol)
    assert sha(source_path) == source_hash
    dump(destination, {'passed': True, 'manifest_sha256': digest, 'cases': cases,
        'preregistration_sha256': sha(prereg_path), 'engineering_protocol_sha256': sha(OUT/'protocol.json'),
        'fit_only_indices': ids.tolist(), 'source_checkpoint_sha256': {str(source_path): source_hash},
        'code_sha256': protocol['code_sha256'], 'input_artifact_sha256': protocol['input_artifact_sha256'],
        'classification_fits': 0, 'classification_scores': 0, 'new_backbone_view_forwards': 0,
        'engineering_optimizer_steps': 4, 'external_images_read': 0, 'seconds': time.time()-started,
        'scope': 'Four fitting-only actual256-image single-task engineering checks including one optimizer step each. No supervised training run, checkpoint selection, candidate metrics, fresh patient confirmation or generalization claim.'})


if __name__ == '__main__':
    main()
