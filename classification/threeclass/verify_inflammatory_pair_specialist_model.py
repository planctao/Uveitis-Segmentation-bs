"""Full-batch conditional mass invariants and independent complete-task gradients."""
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
from inflammatory_pair_specialist import InflammatoryPairSpecialistModel, pair_correction
from inflammatory_pair_specialist_witness import tensors_for, weights_for, write_witness, audit_witness, DETAILS
from inflammatory_pair_specialist_experiments import restore
from audit_inflammatory_pair_specialist import independent_probability
from audit_normal_tangent import complete_task


def main():
    torch.set_num_threads(4)
    started = time.time()
    # The mathematically constrained path also handles zero/tiny abnormal mass.
    edge = torch.tensor([[0., 0., 1.], [0., 1., 0.], [1., 0., 0.],
        [1e-30, .5, .5], [.49, 1e-30, .51], [.2, .3, .5]], device='cuda')
    for value in [-1000., -1., 0., 1., 1000.]:
        corrected, shift, transfer = pair_correction(edge, torch.full((len(edge),), value, device='cuda'))
        assert torch.isfinite(corrected).all() and (corrected >= 0).all()
        assert torch.equal(corrected[:, 2], edge[:, 2])
        assert torch.allclose(corrected.sum(1), edge.sum(1), atol=1e-7, rtol=1e-7)
        if value == 0.:
            assert torch.equal(corrected, edge) and torch.count_nonzero(transfer) == 0
    frame, features, bank, tail, digest = load_all()
    fit = np.flatnonzero(frame.split.eq('train'))
    ids = fit[np.random.default_rng(20292700).permutation(len(fit))[:256]]
    source_path = ROOT/'runs/large_token_evidence/registers/validation/seed_0/model.pt'
    source_hash = sha(source_path)
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    plan_path = ROOT/'results/inflammatory_pair_specialist_preregistration_before_value_joint_gated_readout_outcomes.json'
    planned = json.loads(plan_path.read_text())
    assert planned['classification_fits'] == planned['scores_evaluated'] == 0
    cases = []
    for arm in planned['configuration_plan']:
        config = {key: arm[key] for key in ['evidence', 'hidden']}
        model = InflammatoryPairSpecialistModel(source, tail, config, 20292700).cuda().train()
        names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        assert all(name.startswith('specialist.') for name in names)
        assert len(names) == (4 if config['hidden'] else 2)
        frozen = {name: array_sha(value.cpu().numpy()) for name, value in model.state_dict().items() if name not in names}
        base, selected, registers = tensors_for(features, bank, ids, source, 'cuda')
        y, disease, weights = weights_for(frame, fit, ids, source, 'cuda')
        with torch.no_grad():
            zero = model(base, selected, registers)
            direct = model.head({**base, 'side_tokens': ((registers-model.source_mean_side_tokens)/model.source_std_side_tokens).clamp(-10, 10)})
            for key in ['p', 'subtype', 'embedding']:
                assert torch.equal(zero[key], direct[key])
            assert torch.count_nonzero(zero['pair_transferred_mass']) == 0
        with tempfile.TemporaryDirectory(prefix='pair-specialist-task-') as folder:
            path = Path(folder)/'first_task.npz'
            write_witness(path, model, source, features, bank, frame, fit, ids, config, 0)
            witness = audit_witness(path, source, tail, features, bank, frame, fit, config, 0)
        with torch.no_grad():
            for parameter in model.specialist[-1].parameters():
                parameter.copy_(torch.sin(torch.arange(parameter.numel(), device='cuda').reshape_as(parameter)*.17)*.01)
        torch.cuda.synchronize()
        begin = time.time()
        out = model(base, selected, registers)
        task = loss_fn(out, y, disease, weights, source['cfg'])
        task.backward()
        torch.cuda.synchronize()
        seconds = time.time()-begin
        assert torch.equal(out['p'][:, 2], direct['p'][:, 2])
        assert torch.equal(out['subtype'], direct['subtype']) and torch.equal(out['embedding'], direct['embedding'])
        assert (out['p'] >= 0).all() and torch.isfinite(out['p']).all()
        assert torch.allclose(out['p'][:, :2].sum(1), direct['p'][:, :2].sum(1), atol=1e-7, rtol=1e-7)
        state = {name: parameter.detach().clone().requires_grad_() for name, parameter in model.named_parameters() if parameter.requires_grad}
        raw = {key: torch.from_numpy(value[ids]).cuda() for key, value in features.items()}
        manual = independent_probability(source, {key: value.cuda() for key, value in tail.state_dict().items()}, state, raw, selected, config)
        manual_loss = complete_task(manual, y, disease, weights, source['cfg'])
        manual_loss.backward()
        errors = {}
        for key in DETAILS:
            assert torch.allclose(out[key], manual[key], atol=1e-5, rtol=1e-5), (key, float((out[key]-manual[key]).abs().max()))
        assert torch.equal(out['p'].argmax(1), manual['p'].argmax(1))
        assert abs(float(task.detach())-float(manual_loss.detach())) < 2e-5
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                errors[name] = float((parameter.grad-state[name].grad).abs().max())
                assert torch.allclose(parameter.grad, state[name].grad, atol=1e-6, rtol=3e-3), (name, errors[name])
                assert torch.isfinite(parameter.grad).all() and torch.count_nonzero(parameter.grad) > 0
            else:
                assert parameter.grad is None
        checkpoint = {'source_checkpoint': str(source_path), 'source_checkpoint_sha256': source_hash,
            'adapter_config': config, 'seed': 0,
            'value_state_dict': {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}}
        with tempfile.TemporaryDirectory(prefix='pair-specialist-restore-') as folder:
            path = Path(folder)/'model.pt'
            torch.save(checkpoint, path)
            restored, _ = restore(torch.load(path, map_location='cpu', weights_only=False), tail)
            with torch.inference_mode():
                replay = restored(base, selected, registers)
            for key in ['p', 'subtype', 'embedding']:
                assert torch.equal(replay[key], out[key])
        torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=.01).step()
        for name, value in model.state_dict().items():
            if name in frozen:
                assert array_sha(value.cpu().numpy()) == frozen[name]
        cases.append({'configuration': config, 'actual_images': len(ids),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'first_task_witness': witness, 'nonzero_specialist_complete_task_gradient_errors': errors,
            'normal_probability_bitwise_unchanged': True, 'abnormal_probability_mass_conserved': True,
            'source_subtype_embedding_unchanged': True, 'zero_specialist_exact_source_probability': True,
            'independent_conditional_ratio_and_source_reconstruction': True,
            'checkpoint_restore_bitwise_identical': True, 'all_frozen_weights_buffers_unchanged_after_optimizer': True,
            'maximum_probability_error': float((out['p']-manual['p']).abs().max()), 'full_batch_forward_backward_seconds': seconds})
        print(json.dumps({'event': 'pair_specialist_preflight', 'config': config, 'gradient_errors': errors}), flush=True)
        del model, restored, out, manual, task, manual_loss, replay, zero, direct, state
        torch.cuda.empty_cache()
    assert sha(source_path) == source_hash
    code = ['inflammatory_pair_specialist.py', 'audit_inflammatory_pair_specialist.py',
        'inflammatory_pair_specialist_witness.py', 'inflammatory_pair_specialist_experiments.py',
        'verify_inflammatory_pair_specialist_model.py', 'patch_adapter_cuda_ready.py']
    dump(ROOT/'verification_inflammatory_pair_specialist_model.json', {'passed': True, 'manifest_sha256': digest,
        'cases': cases, 'edge_case_normal_mass_and_zero_identity': True, 'fit_only_indices': ids.tolist(),
        'source_checkpoint_sha256': {str(source_path): source_hash}, 'preregistration_sha256': sha(plan_path),
        'bank_verification_sha256': sha(ROOT/'verification_register_patch_query_bank.json'),
        'classification_fits': 0, 'classification_scores': 0, 'external_images_read': 0,
        'code_sha256': {name: sha(ROOT/name) for name in code}, 'seconds': time.time()-started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
    print('Four actual256image conditional-specialist tasks and normal-mass invariants passed; zero fits/scores.', flush=True)


if __name__ == '__main__':
    main()
