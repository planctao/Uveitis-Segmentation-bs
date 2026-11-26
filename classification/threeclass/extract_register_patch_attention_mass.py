"""Full-set float64 normalization for existing selected patch keys, development only."""
import fcntl
import json
import os
from pathlib import Path
import time
import traceback
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from timm.layers import apply_rot_embed_cat
from experiment import ROOT, dump
from extract_dinov3_large import Images, make_model
from register_patch_query_features import BANK_KEYS, extract_view, array_sha
from kernel_residual_experiments import sha

OUT = ROOT/'runs/register_patch_attention_mass_bank'
SHAPE = (2091, 2, 16, 4)
NAMES = ['selected_mass', 'prefix_mass']


def verify(protocol):
    assert json.loads((OUT/'protocol.json').read_text()) == protocol
    for section in ['code_sha256', 'frozen_artifacts', 'cache_image_sha256']:
        for name, expected in protocol[section].items():
            assert sha(name) == expected, name


@torch.inference_mode()
def extract_mass(model, tensor, ids, view, expected_bank):
    captured = {}
    hook = model.blocks[-1].attn.register_forward_pre_hook(
        lambda module, args, kwargs: captured.update(x=args[0], rope=kwargs.get('rope')),
        with_kwargs=True)
    try:
        bank, _ = extract_view(model, tensor)
    finally:
        hook.remove()
    for name in BANK_KEYS:
        assert np.array_equal(bank[name], np.asarray(expected_bank[name][ids, view])), (name, ids.tolist(), view)
    module = model.blocks[-1].attn
    assert isinstance(module.q_norm, torch.nn.Identity) and isinstance(module.k_norm, torch.nn.Identity)
    assert model.num_prefix_tokens == 5 and module.num_heads == 16
    x = captured['x']
    batch, sequence, width = x.shape
    assert width == 1024
    with torch.autocast('cuda', dtype=torch.float16):
        bias = torch.cat([module.q_bias, module.k_bias, module.v_bias]) if module.q_bias is not None else module.qkv.bias
        qkv = F.linear(x, module.qkv.weight, bias).reshape(batch, sequence, 3, 16, 64).permute(2, 0, 3, 1, 4)
        query, keys, values = qkv.unbind(0)
        if captured['rope'] is not None:
            keys = torch.cat([keys[:, :, :5], apply_rot_embed_cat(keys[:, :, 5:],
                captured['rope'], half=getattr(module, 'rotate_half', False))], 2).type_as(values)
        query = query[:, :, 1:5]
    assert np.array_equal(query.cpu().numpy().astype('float16'), bank['query_raw'])
    positions = torch.from_numpy(bank['patch_indices'].astype('int64')).cuda()
    selected_keys = keys[:, :, 5:].gather(2, positions[..., None].expand(-1, -1, -1, 64))
    assert np.array_equal(selected_keys.cpu().numpy().astype('float16'), bank['patch_keys'])
    # Match the existing adapter's double-precision Q/K arithmetic, including the full prefix denominator.
    logits = query.double()@keys.double().transpose(-2, -1)/8
    probability = logits.softmax(-1)
    gather = positions[:, :, None, :].expand(-1, -1, 4, -1)+5
    selected = probability.gather(-1, gather)
    mass = selected.sum(-1)
    prefix = probability[:, :, :, :5].sum(-1)
    conditional = ((query.double()@selected_keys.double().transpose(-2, -1))/8).softmax(-1)
    conditional_error = float((conditional*mass[..., None]-selected).abs().amax())
    assert torch.allclose(conditional*mass[..., None], selected, atol=1e-12, rtol=1e-11)
    assert torch.all(mass > 0) and torch.all(mass+prefix <= 1+1e-12)
    result = {'selected_mass': mass.cpu().numpy(), 'prefix_mass': prefix.cpu().numpy()}
    witness = {'query': query.cpu().numpy().astype('float16'),
        'all_keys': keys.cpu().numpy().astype('float16'), 'positions': bank['patch_indices'], **result}
    return result, witness, conditional_error


def independent_witness(witness):
    query, keys = witness['query'].astype('float64'), witness['all_keys'].astype('float64')
    logits = np.matmul(query, keys.swapaxes(-2, -1))/8
    numerator = np.exp(logits-logits.max(-1, keepdims=True))
    p = numerator/numerator.sum(-1, keepdims=True)
    positions = np.broadcast_to(witness['positions'][:, :, None, :], p.shape[:3]+(64,))+5
    selected = np.take_along_axis(p, positions, axis=-1).sum(-1)
    prefix = p[:, :, :, :5].sum(-1)
    error = max(float(np.abs(selected-witness['selected_mass']).max()),
        float(np.abs(prefix-witness['prefix_mass']).max()))
    assert np.allclose(selected, witness['selected_mass'], atol=5e-12, rtol=5e-12)
    assert np.allclose(prefix, witness['prefix_mass'], atol=5e-12, rtol=5e-12)
    return error


def main_run():
    protocol = json.loads((OUT/'protocol.json').read_text())
    assert not (OUT/'verification.json').exists()
    assert all(not (OUT/(name+'.npy')).exists() for name in NAMES), 'Version any recovery; do not overwrite arrays.'
    verify(protocol)
    started = time.time()
    dump(OUT/'state.json', {'status': 'loading', 'pid': os.getpid(), 'external_images_read': 0})
    frame = pd.read_csv(ROOT/'metadata/manifest.csv')
    ids = np.flatnonzero(frame.split.isin(['train', 'val']))
    assert np.array_equal(ids, np.arange(2091))
    source_protocol = json.loads((ROOT/'runs/dinov3_large/protocol.json').read_text())
    model = make_model(source_protocol)
    bank = {name: np.load(ROOT/'runs/register_patch_query_bank/arrays'/(name+'.npy'), mmap_mode='r') for name in BANK_KEYS}
    arrays = {name: np.lib.format.open_memmap(OUT/(name+'.npy'), mode='w+', dtype='float64', shape=SHAPE) for name in NAMES}
    loader = DataLoader(Images(frame, ids), batch_size=8, num_workers=2, pin_memory=True)
    maximum_conditional_error = 0.
    witness_rows = []
    view_forwards = 0
    for selected_ids, a, b in loader:
        chosen = selected_ids.numpy()
        begin = int(chosen[0])
        assert begin % 8 == 0 and np.array_equal(chosen, ids[begin:begin+8])
        views = []
        for view, tensor in enumerate([a, b]):
            measured, witness, error = extract_mass(model, tensor.cuda(non_blocking=True), chosen, view, bank)
            maximum_conditional_error = max(maximum_conditional_error, error)
            views.append(measured)
            view_forwards += 1
            if begin in protocol['independent_witness_batch_starts']:
                reference_error = independent_witness(witness)
                path = OUT/'witnesses'/f'{begin:05d}_{view}.npz'
                path.parent.mkdir(exist_ok=True)
                with path.open('xb') as stream:
                    np.savez_compressed(stream, indices=chosen, **witness)
                witness_rows.append({'indices': chosen.tolist(), 'view': view,
                    'path': str(path), 'sha256': sha(path), 'independent_cpu_mass_maximum_error': reference_error})
        for name in NAMES:
            arrays[name][chosen] = np.stack([row[name] for row in views], 1)
            arrays[name].flush()
        dump(OUT/'chunks'/f'{begin:05d}.json', {'indices': chosen.tolist(),
            'array_sha256': {name: array_sha(arrays[name][chosen]) for name in NAMES},
            'all_six_source_bank_arrays_bitwise_equal': True})
        dump(OUT/'state.json', {'status': 'extracting', 'pid': os.getpid(),
            'completed_images': begin+len(chosen), 'total_images': 2091, 'external_images_read': 0})
        if begin % 80 == 0:
            print(json.dumps({'event': 'attention_mass_bank_progress', 'images': begin+len(chosen),
                'total': 2091, 'seconds': time.time()-started}), flush=True)
    assert view_forwards == 524 and len(witness_rows) == 6
    for name, value in arrays.items():
        assert value.shape == SHAPE and value.dtype == np.dtype('float64') and np.isfinite(value).all()
    assert (arrays['selected_mass'] > 0).all()
    assert (arrays['selected_mass']+arrays['prefix_mass'] <= 1+1e-12).all()
    for begin in range(0, 2091, 8):
        row = json.loads((OUT/'chunks'/f'{begin:05d}.json').read_text())
        chosen = np.asarray(row['indices'])
        for name in NAMES:
            assert array_sha(arrays[name][chosen]) == row['array_sha256'][name]
    # Fresh source forwards after completing all arrays, with original batch shapes.
    replay_ids = []
    for begin in protocol['independent_witness_batch_starts']:
        chosen = ids[begin:begin+8]
        _, a, b = next(iter(DataLoader(Images(frame, chosen), batch_size=8)))
        for view, tensor in enumerate([a, b]):
            measured, _, error = extract_mass(model, tensor.cuda(), chosen, view, bank)
            for name in NAMES:
                assert np.array_equal(measured[name], arrays[name][chosen, view]), (begin, view, name)
            view_forwards += 1
        replay_ids.append(chosen.tolist())
    assert view_forwards == 530
    verify(protocol)
    frozen = {str(OUT/(name+'.npy')): sha(OUT/(name+'.npy')) for name in NAMES}
    frozen.update({str(path): sha(path) for folder in ['chunks', 'witnesses'] for path in (OUT/folder).iterdir()})
    dump(OUT/'verification.json', {'passed': True, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': sha(OUT/'protocol.json'), 'manifest_sha256': protocol['manifest_sha256'],
        'development_images': 2091, 'shape': list(SHAPE), 'dtype': 'float64',
        'all_six_original_bank_arrays_reproduced_bitwise_for_every_image': True,
        'normalization': protocol['normalization'], 'maximum_conditional_times_mass_error': maximum_conditional_error,
        'independent_cpu_witnesses': witness_rows, 'fresh_bitwise_mass_replay_batches': replay_ids,
        'frozen_artifacts': frozen, 'classification_fits': 0, 'classification_scores': 0,
        'classifier_probability_predictions': 0, 'backbone_view_forwards': view_forwards,
        'external_images_read': 0, 'seconds': time.time()-started,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()})
    dump(OUT/'state.json', {'status': 'completed', 'pid': os.getpid(), 'development_images': 2091,
        'classification_fits': 0, 'classification_scores': 0, 'external_images_read': 0})
    print('Development-only full-normalization mass bank completed and independently checked.', flush=True)


def main():
    torch.set_num_threads(4)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            main_run()
        except BaseException:
            dump(OUT/'state.json', {'status': 'failed', 'pid': os.getpid(), 'error': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()
