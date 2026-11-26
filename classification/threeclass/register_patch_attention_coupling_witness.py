"""Actual first-task tensors and complete gradient replay for reader coupling."""
import json
import numpy as np
import torch
from experiment import loss_fn
from audit_normal_tangent import complete_task
from register_patch_query_selection_witness import tensors_for, weights_for, DETAILS as PATCH_DETAILS
from register_patch_query_selection_data import INPUT_KEYS
from register_patch_query_features import array_sha
from register_patch_attention_coupling import RegisterPatchAttentionCouplingModel, reconstruction_state, ATTENTION_PREFIX, PATCH_NAMES
from audit_register_patch_attention_coupling import independent_probability

DETAILS = PATCH_DETAILS+['token_query', 'token_keys', 'token_values', 'token_attention',
    'token_pooled', 'token_delta', 'token_base_hidden', 'token_projected']


def write_witness(path, model, source, features, bank, frame, fit, ids, config, seed):
    device = next(model.parameters()).device
    base, selected, registers = tensors_for(features, bank, ids, source, device)
    y, disease, weights = weights_for(frame, fit, ids, source, device)
    output = model(base, selected, registers)
    task = loss_fn(output, y, disease, weights, source['cfg'])
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    gradient = torch.autograd.grad(task, [p for _, p in parameters])
    proof = {'indices': np.asarray(ids), 'fit_indices': np.asarray(fit),
        'seed': np.array(seed), 'config': np.array(json.dumps(config, sort_keys=True)),
        'loss': np.array(float(task.detach())), 'trainable_parameter_names': np.asarray([n for n, _ in parameters]),
        'gradient': torch.cat([g.flatten() for g in gradient]).detach().cpu().numpy(),
        'y': y.cpu().numpy(), 'disease': disease.cpu().numpy(),
        'probability': output['p'].detach().cpu().numpy()}
    for key, value in weights.items():
        proof['weight_'+key] = value.cpu().numpy()
    for name, parameter in parameters:
        proof['initial_sha256_'+name] = np.array(array_sha(parameter.detach().cpu().numpy()))
    for key in DETAILS:
        proof['output_sha256_'+key] = np.array(array_sha(output[key].detach().cpu().numpy()))
    for key in INPUT_KEYS:
        proof['bank_sha256_'+key] = np.array(array_sha(np.asarray(bank[key][ids])))
    for key in features:
        proof['raw_sha256_'+key] = np.array(array_sha(features[key][ids]))
    np.savez_compressed(path, **proof)


def audit_witness(path, source, tail, features, bank, frame, fit, config, seed):
    with np.load(path) as archive:
        proof = {key: archive[key] for key in archive.files}
    assert json.loads(str(proof['config'])) == config and int(proof['seed']) == seed
    assert np.array_equal(proof['fit_indices'], fit)
    ids = fit[np.random.default_rng(20292900+seed).permutation(len(fit))[:256]]
    assert np.array_equal(ids, proof['indices'])
    for key in INPUT_KEYS:
        assert str(proof['bank_sha256_'+key]) == array_sha(np.asarray(bank[key][ids]))
    for key in features:
        assert str(proof['raw_sha256_'+key]) == array_sha(features[key][ids])
    model = RegisterPatchAttentionCouplingModel(source, tail, config, 20292900+seed).cuda().train()
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert list(parameters) == proof['trainable_parameter_names'].tolist()
    for name, parameter in parameters.items():
        assert str(proof['initial_sha256_'+name]) == array_sha(parameter.detach().cpu().numpy())
        assert name in PATCH_NAMES or (name.startswith(ATTENTION_PREFIX) and name != ATTENTION_PREFIX+'key.bias')
    base, selected, registers = tensors_for(features, bank, ids, source, 'cuda')
    output = model(base, selected, registers)
    for key in DETAILS:
        assert str(proof['output_sha256_'+key]) == array_sha(output[key].detach().cpu().numpy()), key
    raw = {key: torch.from_numpy(value[ids]).cuda() for key, value in features.items()}
    state = reconstruction_state(model, differentiable=True)
    manual = independent_probability(source, model.tail.state_dict(), state, raw, selected, config)
    assert torch.allclose(output['p'], manual['p'], atol=5e-5, rtol=5e-5)
    assert torch.equal(output['p'].argmax(1), manual['p'].argmax(1))
    y, disease, weights = weights_for(frame, fit, ids, source, 'cuda')
    assert np.array_equal(y.cpu().numpy(), proof['y']) and np.array_equal(disease.cpu().numpy(), proof['disease'])
    for key, value in weights.items():
        assert np.array_equal(value.cpu().numpy(), proof['weight_'+key])
    task = complete_task(manual, y, disease, weights, source['cfg'])
    assert abs(float(task.detach())-float(proof['loss'])) < 2e-5
    gradients = torch.autograd.grad(task, [state[name] for name in parameters])
    gradient = torch.cat([g.flatten() for g in gradients]).detach().cpu().numpy()
    error = float(np.abs(gradient-proof['gradient']).max())
    assert np.allclose(gradient, proof['gradient'], atol=1e-6, rtol=3e-3), error
    assert not any(p.requires_grad for p in model.tail.parameters())
    assert all(not p.requires_grad for name, p in model.head.named_parameters()
        if not name.startswith('token_evidence.') or name == 'token_evidence.key.bias')
    return {'actual_images': len(ids), 'trainable_parameter_names': list(parameters),
        'trainable_parameters': sum(p.numel() for p in parameters.values()),
        'complete_task_gradient_error': error, 'original_cuda_outputs_sha256_restored': True,
        'frozen_feature_bank_inputs_sha256_restored': True,
        'only_attention_except_constant_key_bias_and_planned_patch_parameters_trainable': True,
        'frozen_base_classifier_normal_geometry_and_tail_verified': True}
