"""Describe full-attention mass of existing top64 positions on fixed development images."""
import fcntl
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import timm
from timm.layers import apply_rot_embed_cat
from torchvision import transforms as T
from safetensors.torch import load_file
from experiment import ROOT, dump
from kernel_residual_experiments import sha
from register_patch_query_features import extract_view
from research_patch_image_inference import FrozenPatchImageExtractor

OUT = ROOT/'runs/selected_attention_mass_diagnostic'
DEST = ROOT/'results/selected_attention_mass_diagnostic'


def verify(protocol):
    for section in ['code_sha256', 'frozen_artifacts', 'image_sha256']:
        for name, expected in protocol[section].items():
            assert sha(name) == expected, name


@torch.inference_mode()
def measure(model, tensor, expected_indices, tolerance):
    captured = {}
    handle = model.blocks[-1].attn.register_forward_pre_hook(
        lambda module, args, kwargs: captured.update(x=args[0], rope=kwargs.get('rope')),
        with_kwargs=True)
    try:
        bank, _ = extract_view(model, tensor)
    finally:
        handle.remove()
    assert np.array_equal(bank['patch_indices'], expected_indices)
    attention = model.blocks[-1].attn
    x = captured['x']
    batch, sequence, width = x.shape
    assert width == 1024 and model.num_prefix_tokens == 5 and attention.num_heads == 16
    assert isinstance(attention.q_norm, torch.nn.Identity)
    assert isinstance(attention.k_norm, torch.nn.Identity)
    with torch.autocast('cuda', dtype=torch.float16):
        bias = torch.cat((attention.q_bias, attention.k_bias, attention.v_bias)) if attention.q_bias is not None else attention.qkv.bias
        qkv = F.linear(x, attention.qkv.weight, bias).reshape(batch, sequence, 3, 16, 64).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        if captured['rope'] is not None:
            key = torch.cat([key[:, :, :5], apply_rot_embed_cat(
                key[:, :, 5:], captured['rope'], half=getattr(attention, 'rotate_half', False))], 2).type_as(value)
        query = query[:, :, 1:5]
        logits = query@key.transpose(-2, -1)/8
        probability = logits.softmax(-1)
        importance = probability.mean(2)[:, :, 5:]
        positions = torch.argsort(importance, dim=-1, descending=True, stable=True)[:, :, :64]
        assert np.array_equal(positions.cpu().numpy(), expected_indices)
    assert np.array_equal(query.cpu().numpy().astype('float16'), bank['query_raw'])
    gather_vectors = positions[..., None].expand(-1, -1, -1, 64)
    assert np.array_equal(key[:, :, 5:].gather(2, gather_vectors).cpu().numpy().astype('float16'), bank['patch_keys'])
    # Float64 reference starts from the same rounded logits; it does not assert full-fp64 backbone equivalence.
    shifted = logits.double()-logits.double().amax(-1, keepdim=True)
    numerator = shifted.exp()
    independent = numerator/numerator.sum(-1, keepdim=True)
    error = float((probability.double()-independent).abs().amax())
    assert error <= tolerance
    gather = positions[:, :, None, :].expand(-1, -1, 4, -1)
    full = probability.double()
    selected = full[:, :, :, 5:].gather(-1, gather).sum(-1)
    spatial = full[:, :, :, 5:].sum(-1)
    prefix = full[:, :, :, :5].sum(-1)
    omitted = spatial-selected
    assert torch.all(selected > 0) and torch.all(omitted >= -1e-12)
    assert torch.allclose(prefix+spatial, torch.ones_like(spatial), atol=1e-3, rtol=0)
    independent_selected = independent[:, :, :, 5:].gather(-1, gather).sum(-1)
    assert torch.allclose(selected, independent_selected, atol=1e-3, rtol=1e-3)
    data = {'selected_mass': selected, 'spatial_mass': spatial, 'prefix_mass': prefix,
        'omitted_spatial_mass': omitted, 'fraction_of_spatial_mass_selected': selected/spatial,
        'conditional_renormalization_multiplier': 1/selected}
    assert all(torch.isfinite(value).all() for value in data.values())
    return {name: value.cpu().numpy() for name, value in data.items()}, error


def main_run():
    started = time.time()
    protocol = json.loads((OUT/'protocol.json').read_text())
    assert not (DEST/'analysis.json').exists() and not (DEST/'measurements.npz').exists()
    verify(protocol)
    dump(OUT/'state.json', {'status': 'loading', 'pid': os.getpid(), 'external_test_evaluated': False})
    frame = pd.read_csv(ROOT/'metadata/manifest.csv')
    weight = protocol['pretrained_weights']
    model = timm.create_model('vit_large_patch16_dinov3_qkvb', pretrained=False,
        num_classes=0, img_size=320, dynamic_img_size=True)
    model.load_state_dict(load_file(weight['storage'], device='cpu'), strict=True)
    model.eval().cuda()
    assert len(model.blocks) == 24 and model.num_features == 1024
    expected = np.load(ROOT/'runs/register_patch_query_bank/arrays/patch_indices.npy', mmap_mode='r')
    arrays, errors, indices = {}, [], []
    for batch in protocol['development_batches']:
        ids = np.asarray(batch, dtype='int64')
        assert len(ids) == 8 and ids[0] % 8 == 0 and np.array_equal(ids, np.arange(ids[0], ids[0]+8))
        assert frame.iloc[ids].split.isin(['train', 'val']).all()
        images = FrozenPatchImageExtractor.prepare_images(frame.iloc[ids].path.tolist())
        indices.extend(ids.tolist())
        per_view = []
        for view, size in enumerate([320, 448]):
            dump(OUT/'state.json', {'status': 'measuring', 'pid': os.getpid(),
                'manifest_indices': ids.tolist(), 'size': size, 'external_test_evaluated': False})
            transform = T.Compose([T.Resize((size, size), antialias=True), T.ToTensor(),
                T.Normalize((.485, .456, .406), (.229, .224, .225))])
            tensor = torch.stack([transform(image) for image in images]).cuda()
            data, error = measure(model, tensor, np.asarray(expected[ids, view]),
                protocol['rounded_logit_probability_absolute_tolerance'])
            per_view.append(data)
            errors.append({'manifest_indices': ids.tolist(), 'size': size, 'maximum_probability_error': error})
        for name in per_view[0]:
            arrays.setdefault(name, []).append(np.stack([row[name] for row in per_view], 1))
        print(json.dumps({'event': 'attention_mass_batch_measured', 'indices': ids.tolist()}), flush=True)
    assert len(indices) == len(set(indices)) == 24
    values = {name: np.concatenate(parts, 0) for name, parts in arrays.items()}
    assert all(value.shape == (24, 2, 16, 4) for value in values.values())
    summary = {}
    for view, size in enumerate([320, 448]):
        summary[str(size)] = {}
        for name, array in values.items():
            flat = array[:, view].reshape(-1)
            summary[str(size)][name] = {'mean': float(flat.mean()), 'minimum': float(flat.min()),
                'median': float(np.median(flat)), 'p05': float(np.quantile(flat, .05)),
                'p95': float(np.quantile(flat, .95)), 'maximum': float(flat.max())}
    verify(protocol)
    DEST.mkdir(exist_ok=True)
    with (DEST/'measurements.npz').open('xb') as stream:
        np.savez_compressed(stream, manifest_indices=np.asarray(indices), **values)
    result = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'passed': True, 'scope': protocol['scope'], 'limitations': protocol['limitations'],
        'protocol_sha256': sha(OUT/'protocol.json'), 'code_sha256': protocol['code_sha256'],
        'development_images': 24, 'backbone_view_batches': 6, 'sizes': [320, 448],
        'head_query_observations_per_resolution': 24*16*4,
        'original_batch_top64_positions_query_and_selected_keys_bitwise_verified': True,
        'rounded_logit_reference_checks': errors, 'summary': summary,
        'measurements_sha256': sha(DEST/'measurements.npz'),
        'new_model_fits': 0, 'new_classifier_probability_predictions': 0, 'new_candidate_scores': 0,
        'backbone_forward_batches': 6, 'external_images_read': 0,
        'external_test_evaluated': False, 'seconds': time.time()-started,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()}
    with (DEST/'analysis.json').open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    dump(OUT/'state.json', {'status': 'completed', 'pid': os.getpid(), 'passed': True,
        'external_test_evaluated': False})
    print(json.dumps({'passed': True, 'summary': summary}), flush=True)


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
