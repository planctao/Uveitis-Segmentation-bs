"""Actual 256-image task, gradients, anchors and restoration for all three arms."""
import hashlib
import json
import os
import tempfile
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from experiment import ROOT, KEYS, normalize, loss_fn, dump
from audit_normal_tangent import complete_task
from register_multiquery_data import load_all
from register_patch_query_witness import weights_for
from large_lastblock_delta_lora import make_templates
from large_lastblock_delta_lora_cache_v2 import build_cache, array_sha
from large_lastblock_readout_coupling import ARMS, SEED_BASE, LastBlockReadoutCoupling, restore

OUT = ROOT/'runs/large_lastblock_readout_coupling_model_preflight'
DEST = ROOT/'verification_large_lastblock_readout_coupling_model.json'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(protocol):
    for section in ['code_sha256', 'input_artifact_sha256', 'package_implementation_sha256']:
        for name, expected in protocol[section].items():
            path = Path(name)
            assert sha(path if path.is_absolute() else ROOT/path) == expected, name


def gradients(model, cache, features, ids, y, disease, weights, cfg,
              independent=False, checkpointed=True):
    model.zero_grad(set_to_none=True)
    out = model(cache, features, ids, independent=independent, use_checkpoint=checkpointed)
    task = (complete_task if independent else loss_fn)(out, y, disease, weights, cfg)
    assert torch.isfinite(task)
    task.backward()
    gradient = {name: parameter.grad.detach().clone()
                for name, parameter in model.trainable_parameters().items()}
    assert all(torch.isfinite(value).all() for value in gradient.values())
    return float(task.detach()), {key: out[key].detach().clone()
                                 for key in ['p', 'subtype', 'embedding', 'fine_embedding']}, gradient


def compare(first, second, atol=1e-6, rtol=3e-3):
    assert first.keys() == second.keys()
    errors = {}
    for name in first:
        error = float((first[name]-second[name]).abs().max())
        assert torch.allclose(first[name], second[name], atol=atol, rtol=rtol), (name, error)
        errors[name] = error
    return errors


def main():
    assert not DEST.exists(), 'Preserve completed proof; version any recovery.'
    started = time.time()
    torch.set_num_threads(4)
    protocol = json.loads((OUT/'protocol.json').read_text())
    verify(protocol)
    frame, features, digest = load_all()
    fit = np.flatnonzero(frame.split.eq('train'))
    ids = fit[np.random.default_rng(SEED_BASE).permutation(len(fit))[:256]]
    assert ids.tolist() == protocol['indices'] and frame.iloc[ids].split.eq('train').all()
    parent = json.loads((ROOT/'runs/dinov3_large/protocol.json').read_text())
    cache, metadata = build_cache(frame, ids, parent)
    templates = make_templates(parent)
    source_path = ROOT/'runs/large_token_evidence/registers/validation/seed_0/model.pt'
    source_hash = sha(source_path)
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    y, disease, weights = weights_for(frame, fit, ids, source, 'cuda')
    positive = ((disease[:, None] == disease[None, :]) &
                (weights['patient_ids'][:, None] != weights['patient_ids'][None, :]))
    chunk = torch.arange(len(ids), device='cuda')//8
    across = int((positive & (chunk[:, None] != chunk[None, :])).sum())
    assert across > 0
    print(json.dumps({'event': 'actual256_cache_ready', 'cross_microbatch_positive_pairs': across}), flush=True)
    cases = []
    for arm in ARMS:
        dump(OUT/'state.json', {'status': 'running_actual256_model_engineering', 'pid': os.getpid(),
                              'arm': arm, 'classification_fits': 0, 'classification_scores': 0})
        model = LastBlockReadoutCoupling(source, templates, arm, 0).cuda().train()
        parameters = model.trainable_parameters()
        readout_count = 257*len(source['diseases'])
        count = readout_count + (0 if arm == 'readout_only' else 16384)
        assert sum(parameter.numel() for parameter in parameters.values()) == count
        frozen = {name: array_sha(value.detach().cpu().numpy())
                  for name, value in model.state_dict().items() if name not in parameters}
        normalized, _ = normalize({key: features[key][ids] for key in list(KEYS)+['side_tokens']},
                                  None, source['stats'])
        with torch.no_grad():
            anchored = model(cache, features, ids, return_raw=True)
            direct = model.head({key: torch.from_numpy(value).cuda() for key, value in normalized.items()})
            for key in list(KEYS)+['side_tokens']:
                assert np.array_equal(anchored['_anchored_raw'][key].cpu().numpy(), features[key][ids]), key
            for key in ['p', 'subtype', 'embedding', 'fine_embedding']:
                assert torch.equal(anchored[key], direct[key]), key
            assert torch.equal(loss_fn(anchored, y, disease, weights, source['cfg']),
                               loss_fn(direct, y, disease, weights, source['cfg']))
        loss, original, gradient = gradients(model, cache, features, ids, y, disease, weights, source['cfg'])
        assert gradient['head.flat.weight'].abs().max() > 0 and gradient['head.flat.bias'].abs().max() > 0
        if arm != 'readout_only':
            assert torch.count_nonzero(gradient['adapter_a']) == 0 and gradient['adapter_b'].abs().max() > 0
        manual_loss, manual, manual_gradient = gradients(model, cache, features, ids, y, disease,
                                                         weights, source['cfg'], independent=True)
        assert abs(loss-manual_loss) < 2e-5
        initial_errors = compare(gradient, manual_gradient)
        compare(original, manual, atol=5e-5, rtol=5e-5)
        witness = OUT/'cases'/(arm+'_actual_first_task.npz')
        witness.parent.mkdir(exist_ok=True)
        assert not witness.exists()
        payload = {'indices': ids, 'fit_indices': fit, 'arm': np.array(arm), 'seed': np.array(0),
                   'loss': np.array(loss), 'parameter_names': np.asarray(list(parameters)),
                   'gradient': torch.cat([gradient[name].flatten() for name in parameters]).cpu().numpy(),
                   'p': original['p'].cpu().numpy()}
        for name, parameter in parameters.items():
            payload['initial_sha256_'+name] = np.array(array_sha(parameter.detach().cpu().numpy()))
        for key in features:
            payload['raw_sha256_'+key] = np.array(array_sha(features[key][ids]))
        for size in [320, 448]:
            payload['cache_sha256_'+str(size)] = np.array(metadata['arrays'][str(size)]['sha256'])
        np.savez_compressed(witness, **payload)
        optimizer = torch.optim.AdamW(list(parameters.values()), lr=1e-4, weight_decay=.01)
        for name, parameter in parameters.items():
            parameter.grad = gradient[name].clone()
        torch.nn.utils.clip_grad_norm_(list(parameters.values()), 3.)
        optimizer.step()
        loss, trained, gradient = gradients(model, cache, features, ids, y, disease, weights, source['cfg'])
        assert all(value.abs().max() > 0 for value in gradient.values())
        manual_loss, manual, manual_gradient = gradients(model, cache, features, ids, y, disease,
                                                         weights, source['cfg'], independent=True)
        assert abs(loss-manual_loss) < 2e-5
        updated_errors = compare(gradient, manual_gradient)
        compare(trained, manual, atol=5e-5, rtol=5e-5)
        assert torch.equal(trained['p'].argmax(1), manual['p'].argmax(1))
        small_weights = {key: value[:8] for key, value in weights.items()}
        small_loss, small, small_gradient = gradients(model, cache, features, ids[:8], y[:8], disease[:8],
                                                       small_weights, source['cfg'])
        direct_loss, direct, direct_gradient = gradients(model, cache, features, ids[:8], y[:8], disease[:8],
                                                          small_weights, source['cfg'], checkpointed=False)
        assert abs(small_loss-direct_loss) < 1e-7
        direct_errors = compare(small_gradient, direct_gradient, atol=1e-7, rtol=1e-5)
        compare(small, direct, atol=0, rtol=0)
        checkpoint = {'arm': arm, 'seed': 0, 'source_checkpoint': str(source_path),
                      'source_checkpoint_sha256': source_hash,
                      'trainable_state_dict': {name: parameter.detach().cpu().clone()
                                               for name, parameter in parameters.items()}}
        with tempfile.TemporaryDirectory(prefix='readout-coupling-restore-') as directory:
            path = Path(directory)/'model.pt'
            torch.save(checkpoint, path)
            saved = torch.load(path, map_location='cpu', weights_only=False)
            restored = restore(saved, source, templates)
            with torch.no_grad():
                again = restored(cache, features, ids)
            for key in ['p', 'subtype', 'embedding', 'fine_embedding']:
                assert torch.equal(again[key], trained[key]), key
            del restored, again
        assert sha(source_path) == source_hash
        for name, expected in frozen.items():
            assert array_sha(model.state_dict()[name].detach().cpu().numpy()) == expected, name
        assert all(not parameter.requires_grad for parameter in model.block.parameters())
        assert all(not parameter.requires_grad for name, parameter in model.head.named_parameters()
                   if not name.startswith('flat.'))
        case = {'arm': arm, 'actual_images': 256, 'trainable_parameters': count,
                'trainable_readout_parameters': readout_count, 'adapter_trainable_parameters': count-readout_count,
                'trainable_parameter_names': list(parameters), 'source_zero_raw_output_and_task_bitwise': True,
                'all_readout_gradients_finite_and_nonzero': True,
                'zero_adapter_a_exact_and_b_nonzero': None if arm == 'readout_only' else True,
                'all_updated_trainable_gradients_finite_and_nonzero': True,
                'initial_gradient_errors': initial_errors, 'updated_gradient_errors': updated_errors,
                'actual8_checkpoint_direct_gradient_errors': direct_errors,
                'cross_microbatch_positive_pairs_in_one256_task': across,
                'cuda_restore_bitwise': True, 'frozen_parameters_buffers_and_source_unchanged': True,
                'actual_first_task_witness': str(witness), 'actual_first_task_witness_sha256': sha(witness),
                'engineering_optimizer_steps': 1}
        cases.append(case)
        print(json.dumps({'event': 'actual256_model_case_passed', **case}), flush=True)
        del model, parameters, optimizer, anchored, original, direct, gradient, manual, trained, manual_gradient
        torch.cuda.empty_cache()
    verify(protocol)
    record = {'passed': True, 'created_at_utc': protocol['created_at_utc'], 'seconds': time.time()-started,
              'cases': cases, 'engineering_optimizer_steps': 3, 'classification_fits': 0,
              'classification_scores': 0, 'external_raw_images_read': 0, 'manifest_sha256': digest,
              'cache_metadata': metadata, 'engineering_protocol_sha256': sha(OUT/'protocol.json'),
              'code_sha256': protocol['code_sha256'], 'input_artifact_sha256': protocol['input_artifact_sha256'],
              'package_implementation_sha256': protocol['package_implementation_sha256'],
              'scope': 'Three actual256 zero/update complete tasks/all gradients, source anchors, actual8 checkpoint/direct and CUDA restore. Final block oracle shares stable SDP; readout has separate affine/exp marginalization arithmetic but frozen feature encoder is shared. Three engineering steps are not classifier fits/scores. Engine, plans, gates and formal launch remain required.'}
    dump(DEST, record)
    print('Three actual256 model/task proofs passed; no classifier fits or scores.', flush=True)


if __name__ == '__main__':
    dump(OUT/'state.json', {'status': 'running_actual256_model_engineering', 'pid': os.getpid(),
                          'classification_fits': 0, 'classification_scores': 0})
    try:
        main()
        dump(OUT/'state.json', {'status': 'completed', 'pid': os.getpid(), 'classification_fits': 0,
                              'classification_scores': 0, 'engineering_optimizer_steps': 3})
    except BaseException:
        dump(OUT/'state.json', {'status': 'failed', 'pid': os.getpid(), 'traceback': traceback.format_exc(),
                              'classification_fits': 0, 'classification_scores': 0})
        raise
