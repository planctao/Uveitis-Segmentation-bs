"""Shared source-matched trainer and complete-task witnesses for all three arms."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from experiment import KEYS, dump, normalize, loss_fn, metrics
from register_patch_query_witness import weights_for
from audit_normal_tangent import complete_task
from large_lastblock_delta_lora_cache_v2 import array_sha
from large_lastblock_delta_lora_engine import sha
from large_lastblock_readout_coupling import SEED_BASE, LastBlockReadoutCoupling, restore as restore_model


def first_ids(fit, seed):
    return np.asarray(fit)[np.random.default_rng(SEED_BASE+seed).permutation(len(fit))[:256]]


def frozen_bytes(model):
    trainable = model.trainable_parameters()
    return {name: array_sha(value.detach().cpu().numpy())
            for name, value in model.state_dict().items() if name not in trainable}


@torch.no_grad()
def source_identity(model, source, features, cache, ids):
    anchored = model(cache, features, ids, return_raw=True)
    normalized, _ = normalize({key: features[key][ids] for key in list(KEYS)+['side_tokens']},
                              None, source['stats'])
    direct = model.head({key: torch.from_numpy(value).cuda() for key, value in normalized.items()})
    for key in list(KEYS)+['side_tokens']:
        assert np.array_equal(anchored['_anchored_raw'][key].cpu().numpy(), features[key][ids]), key
    for key in ['p', 'subtype', 'embedding', 'fine_embedding']:
        assert torch.equal(anchored[key], direct[key]), key


def write_witness(path, model, source, features, cache, frame, fit, arm, seed):
    assert not Path(path).exists()
    ids = first_ids(fit, seed)
    assert len(ids) == 256 and model.arm == arm
    source_identity(model, source, features, cache, ids)
    y, disease, weights = weights_for(frame, fit, ids, source, 'cuda')
    model.zero_grad(set_to_none=True)
    out = model(cache, features, ids)
    task = loss_fn(out, y, disease, weights, source['cfg'])
    parameters = model.trainable_parameters()
    gradients = torch.autograd.grad(task, list(parameters.values()))
    by_name = dict(zip(parameters, gradients))
    assert all(torch.isfinite(value).all() for value in gradients)
    assert by_name['head.flat.weight'].abs().max() > 0 and by_name['head.flat.bias'].abs().max() > 0
    if arm != 'readout_only':
        assert torch.count_nonzero(by_name['adapter_a']) == 0 and by_name['adapter_b'].abs().max() > 0
    proof = {'indices': ids, 'fit_indices': np.asarray(fit), 'seed': np.array(seed), 'arm': np.array(arm),
             'loss': np.array(float(task.detach())), 'parameter_names': np.asarray(list(parameters)),
             'gradient': torch.cat([value.flatten() for value in gradients]).cpu().numpy(),
             'probability': out['p'].detach().cpu().numpy()}
    for name, parameter in parameters.items():
        proof['initial_sha256_'+name] = np.array(array_sha(parameter.detach().cpu().numpy()))
    for key in ['p', 'subtype', 'embedding', 'fine_embedding']:
        proof['output_sha256_'+key] = np.array(array_sha(out[key].detach().cpu().numpy()))
    for key in features:
        proof['raw_sha256_'+key] = np.array(array_sha(features[key][ids]))
    locations = {int(index): position for position, index in enumerate(cache['indices'])}
    positions = [locations[int(index)] for index in ids]
    for size in [320, 448]:
        proof['cache_sha256_'+str(size)] = np.array(array_sha(cache['tokens'][size][positions]))
        proof['rope_sha256_'+str(size)] = np.array(array_sha(cache['rope'][size].numpy()))
    np.savez_compressed(path, **proof)


def audit_witness(path, source, templates, features, cache, frame, fit, arm, seed):
    with np.load(path) as saved:
        proof = {key: saved[key] for key in saved.files}
    ids = first_ids(fit, seed)
    assert np.array_equal(ids, proof['indices']) and np.array_equal(fit, proof['fit_indices'])
    assert str(proof['arm']) == arm and int(proof['seed']) == seed
    for key in features:
        assert str(proof['raw_sha256_'+key]) == array_sha(features[key][ids]), key
    locations = {int(index): position for position, index in enumerate(cache['indices'])}
    positions = [locations[int(index)] for index in ids]
    for size in [320, 448]:
        assert str(proof['cache_sha256_'+str(size)]) == array_sha(cache['tokens'][size][positions])
        assert str(proof['rope_sha256_'+str(size)]) == array_sha(cache['rope'][size].numpy())
    model = LastBlockReadoutCoupling(source, templates, arm, seed).cuda().train()
    source_identity(model, source, features, cache, ids)
    parameters = model.trainable_parameters()
    assert list(parameters) == proof['parameter_names'].tolist()
    for name, parameter in parameters.items():
        assert str(proof['initial_sha256_'+name]) == array_sha(parameter.detach().cpu().numpy())
    with torch.no_grad():
        out = model(cache, features, ids)
    for key in ['p', 'subtype', 'embedding', 'fine_embedding']:
        assert str(proof['output_sha256_'+key]) == array_sha(out[key].cpu().numpy()), key
    del out
    y, disease, weights = weights_for(frame, fit, ids, source, 'cuda')
    manual = model(cache, features, ids, independent=True)
    task = complete_task(manual, y, disease, weights, source['cfg'])
    assert abs(float(task.detach())-float(proof['loss'])) < 2e-5
    gradients = torch.autograd.grad(task, list(parameters.values()))
    gradient = torch.cat([value.flatten() for value in gradients]).cpu().numpy()
    error = float(np.abs(gradient-proof['gradient']).max())
    assert np.allclose(gradient, proof['gradient'], atol=1e-6, rtol=3e-3), error
    assert all(torch.isfinite(value).all() for value in gradients)
    expected = 257*len(source['diseases'])+(0 if arm == 'readout_only' else 16384)
    assert sum(parameter.numel() for parameter in parameters.values()) == expected
    by_name = dict(zip(parameters, gradients))
    assert by_name['head.flat.weight'].abs().max() > 0 and by_name['head.flat.bias'].abs().max() > 0
    if arm != 'readout_only':
        assert torch.count_nonzero(by_name['adapter_a']) == 0 and by_name['adapter_b'].abs().max() > 0
    assert not any(parameter.requires_grad for parameter in model.block.parameters())
    assert not any(parameter.requires_grad for name, parameter in model.head.named_parameters()
                   if not name.startswith('flat.'))
    return {'actual_images': 256, 'trainable_parameters': expected, 'trainable_parameter_names': list(parameters),
            'complete_task_gradient_error': error, 'readout_gradients_finite_and_nonzero': True,
            'zero_adapter_a_and_nonzero_b_verified': arm != 'readout_only',
            'actual_source_raw_and_ram_input_hashes_verified': True,
            'source_zero_features_outputs_and_original_cuda_output_hashes_restored': True,
            'only_prescribed_adapter_and_readout_trainable': True}


@torch.no_grad()
def freeze_warmstart(folder, source_path, features, cache, templates, fit, arm, seed):
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    ids = first_ids(fit, seed)
    model = LastBlockReadoutCoupling(source, templates, arm, seed).cuda().eval()
    source_identity(model, source, features, cache, ids)
    path = Path(folder)/'warmstart_freeze_before_adapter.json'
    assert not path.exists()
    dump(path, {'source_checkpoint': str(source_path), 'source_checkpoint_sha256': sha(source_path),
                'arm': arm, 'seed': seed, 'fit_indices': np.asarray(fit).tolist(),
                'first_batch_indices': ids.tolist(), 'actual_images': 256,
                'exact_source_raw_features_probability_subtype_embeddings_identity': True,
                'new_adapter_fits_so_far': 0, 'external_test_evaluated': False})
    del model
    torch.cuda.empty_cache()


@torch.inference_mode()
def predict(model, source, features, cache, ids, independent=False):
    model.cuda().eval()
    answer = [model(cache, features, ids[begin:begin+256], independent=independent)['p'].cpu().numpy()
              for begin in range(0, len(ids), 256)]
    return np.concatenate(answer)


def restore(checkpoint, templates):
    source_path = Path(checkpoint['source_checkpoint'])
    assert sha(source_path) == checkpoint['source_checkpoint_sha256']
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    return restore_model(checkpoint, source, templates), source


def fit_adapter(folder, source_path, templates, features, cache, frame, fit, stop, arm, seed):
    folder = Path(folder)
    assert not (folder/'model.pt').exists()
    folder.mkdir(parents=True, exist_ok=True)
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    model = LastBlockReadoutCoupling(source, templates, arm, seed).cuda().train()
    frozen = frozen_bytes(model)
    write_witness(folder/'training_input_witness.npz', model, source, features, cache, frame, fit, arm, seed)
    parameters = list(model.trainable_parameters().values())
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=.01)
    history, best, waiting = [], None, 0
    generator = np.random.default_rng(SEED_BASE+seed)
    started = time.time()
    for epoch in range(21):
        total = 0.
        model.train()
        if epoch:
            order = fit[generator.permutation(len(fit))]
            for begin in range(0, len(order), 256):
                ids = order[begin:begin+256]
                y, disease, weights = weights_for(frame, fit, ids, source, 'cuda')
                optimizer.zero_grad(set_to_none=True)
                task = loss_fn(model(cache, features, ids), y, disease, weights, source['cfg'])
                assert torch.isfinite(task)
                task.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 3.)
                optimizer.step()
                total += float(task.detach())*len(ids)
        p = predict(model, source, features, cache, stop)
        score = metrics(frame.iloc[stop].label, p)
        key = (score['macro_f1'], -score['log_loss'])
        history.append({'epoch': epoch, 'train_loss': total/len(fit), 'val': score})
        dump(folder/'history.json', history)
        if best is None or key > best:
            best, waiting = key, 0
            checkpoint = {'fit_indices': np.asarray(fit).copy(), 'arm': arm, 'seed': seed,
                          'source_checkpoint': str(source_path), 'source_checkpoint_sha256': sha(source_path),
                          'trainable_state_dict': {name: parameter.detach().cpu().clone()
                                                   for name, parameter in model.trainable_parameters().items()},
                          'epoch': epoch, 'val': score}
            torch.save(checkpoint, folder/'model.pt')
            np.savez_compressed(folder/'val_predictions.npz', p=p)
        else:
            waiting += 1
        print(json.dumps({'event': 'large_lastblock_readout_coupling_epoch', 'folder': str(folder),
                          'epoch': epoch, 'seconds': time.time()-started}), flush=True)
        if waiting >= 5:
            break
    assert frozen == frozen_bytes(model)
    dump(folder/'summary.json', {'arm': arm, 'seed': seed, 'epoch': checkpoint['epoch'], 'val': checkpoint['val'],
                                'seconds': time.time()-started,
                                'trainable_parameters': sum(parameter.numel() for parameter in parameters),
                                'all_other_parameters_and_buffers_unchanged': True,
                                'actual_single_complete_task_batch': 256, 'microbatch': 8, 'new_seed_fits': 1})
    del model
    torch.cuda.empty_cache()
