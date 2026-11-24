"""Frozen B16 block4/block8 spatial regions, with original-input controls."""
import fcntl
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from experiment import ROOT, KEYS, dump
from extract_vit import WEIGHTS
from kernel_residual_experiments import sha
from photometric_data import PhotometricImages
from extract_photometric_views import backbone, array_hash
from early_region_data import load_derived

OUT = ROOT / 'runs/early_regions'
REGIONS = ['regions_block4', 'regions_block8']


@torch.inference_mode()
def regions_for(model, a, b):
    captured = {}; handles = []
    for index in [3, 7]:
        handles.append(model.blocks[index].register_forward_hook(
            lambda module, inputs, output, index=index: captured.update({index: output})))
    early = {k: [] for k in REGIONS}; control = {k: [] for k in KEYS}
    try:
        for size, images in [(320, a), (448, b)]:
            with torch.autocast('cuda', dtype=torch.float16):
                tokens = model.forward_features(images.cuda(non_blocking=True))
                def pool(patches):
                    return F.adaptive_avg_pool2d(patches.transpose(1, 2).reshape(len(images), 768,
                        size // 16, size // 16), (3, 3)).flatten(2).transpose(1, 2)
                for index in [3, 7]:
                    early[f'regions_block{index + 1}'].append(pool(captured[index][:, model.num_prefix_tokens:]).cpu().numpy().astype('float16'))
                patches = tokens[:, model.num_prefix_tokens:]
                values = {'global': tokens[:, 0], 'regions': pool(patches),
                          'mid': captured[7][:, model.num_prefix_tokens:].mean(1),
                          'dispersion': patches.float().std(1)}
            for k, value in values.items():
                control[k].append(value.cpu().numpy().astype('float16'))
    finally:
        for handle in handles:
            handle.remove()
    return ({k: np.concatenate(v, -1) for k, v in early.items()},
            {k: np.concatenate(v, -1) for k, v in control.items()})


def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    digest = sha(ROOT / 'metadata/manifest.csv'); assert digest == protocol['manifest_sha256']
    assert sha(Path(WEIGHTS)) == protocol['pretrained_weights']['sha256']
    frame = pd.read_csv(ROOT / 'metadata/manifest.csv')
    indices = np.flatnonzero(frame.split.isin(['train', 'val']))
    assert np.array_equal(indices, np.arange(2091))
    with np.load(ROOT / 'features/multiscale_vit.npz') as saved:
        assert str(saved['manifest_sha256']) == digest and np.array_equal(saved['path'], frame.path)
        reference = {k: saved[k] for k in KEYS}
    model = backbone(); target = Path(protocol['feature_storage']); target.parent.mkdir(parents=True, exist_ok=True)
    shards = OUT / 'extraction_shards'; shards.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(PhotometricImages(frame, indices, corrected=False), batch_size=16, num_workers=4, pin_memory=True)
    start = time.time(); errors = []
    for step, (actual, a, b) in enumerate(loader):
        chosen = indices[step * 16:(step + 1) * 16]; assert np.array_equal(actual.numpy(), chosen)
        shard = shards / f'{step:05d}.npz'
        if shard.exists():
            with np.load(shard) as saved:
                assert str(saved['manifest_sha256']) == digest and np.array_equal(saved['indices'], chosen)
                assert all(np.isfinite(saved[k]).all() for k in REGIONS)
                error = saved['original_control_relative_mae']
        else:
            values, control = regions_for(model, a, b); error = []
            for key in KEYS:
                expected = reference[key][chosen].astype('float32'); value = control[key].astype('float32')
                error.append(float(np.abs(expected - value).mean() / (np.abs(expected).mean() + 1e-7)))
            error = np.array(error); assert np.max(error) < .003
            with shard.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream, indices=chosen, manifest_sha256=np.array(digest),
                                    original_control_relative_mae=error, **values)
            shard.with_suffix('.tmp').replace(shard)
        errors.append(error); assert np.max(error) < .003
        if step % 10 == 0 or step + 1 == len(loader):
            state = {'status': 'extracting', 'pid': os.getpid(), 'completed_images': int(chosen[-1]) + 1,
                     'development_images': 2091, 'seconds': time.time() - start, 'external_raw_images_read': 0}
            dump(OUT / 'extraction_state.json', state); print(json.dumps(state), flush=True)
    arrays = {k: [] for k in REGIONS}
    for step in range(len(loader)):
        with np.load(shards / f'{step:05d}.npz') as saved:
            assert np.array_equal(saved['indices'], indices[step * 16:(step + 1) * 16])
            for k in REGIONS:
                arrays[k].append(saved[k])
    arrays = {k: np.concatenate(v) for k, v in arrays.items()}
    arrays.update(indices=indices, path=frame.iloc[indices].path.to_numpy(dtype=str), manifest_sha256=np.array(digest))
    if target.exists():
        with np.load(target) as saved:
            assert set(saved.files) == set(arrays) and all(np.array_equal(saved[k], v) for k, v in arrays.items())
    else:
        with target.with_suffix('.tmp').open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        target.with_suffix('.tmp').replace(target)
    dump(target.with_suffix('.json'), {'manifest_sha256': digest, 'weights_sha256': protocol['pretrained_weights']['sha256'],
        'feature_sha256': sha(target), 'development_images': 2091, 'external_raw_images_read': 0,
        'max_original_control_relative_mae': dict(zip(KEYS, np.max(errors, axis=0).tolist())),
        'arrays': {k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'sha256': array_hash(v)} for k, v in arrays.items()}})
    for step in range(len(loader)):
        with np.load(shards / f'{step:05d}.npz') as saved:
            for k in REGIONS:
                assert np.array_equal(saved[k], arrays[k][saved['indices']])
    del model; torch.cuda.empty_cache(); model = backbone(); selected = np.array([0, 352, 887, 1501, 1800, 2090])
    checked = []
    for actual, a, b in DataLoader(PhotometricImages(frame, selected, corrected=False), batch_size=1, num_workers=0):
        row = int(actual[0]); values, _ = regions_for(model, a, b); differences = {}
        for k in REGIONS:
            expected = arrays[k][[row]].astype('float32'); value = values[k].astype('float32')
            relative = float(np.abs(expected - value).mean() / (np.abs(expected).mean() + 1e-7))
            assert relative < .003
            differences[k] = {'relative_mae': relative, 'max_abs_error': float(np.abs(expected - value).max())}
        checked.append({'development_index': row, 'differences': differences})
    for feature in ['block4', 'block8']:
        values = load_derived(ROOT, frame, digest, feature)
        assert all(np.isfinite(v[:2091]).all() and np.isnan(v[2091:]).all() for v in values.values())
    dump(ROOT / 'verification_early_region_features.json', {'passed': True, 'manifest_sha256': digest,
        'weights_sha256': protocol['pretrained_weights']['sha256'], 'development_images': 2091,
        'shards_checked': len(loader), 'actual_images_recomputed': checked, 'external_raw_images_read': 0,
        'checks': ['Pinned B16 weights strictly loaded; every original input reproduces cached M5 feature controls',
            'All intermediate block4/block8 regions and all shard/bank identities exactly aligned',
            'Freshly loaded backbone reproduced six actual images at both scales',
            'Original M5 arrays unchanged; early regions are an extra fit-normalized input only',
            'Both depth layouts reconstructed with all external rows explicitly NaN']})
    dump(OUT / 'extraction_state.json', {'status': 'completed', 'pid': os.getpid(),
        'development_images': 2091, 'verification_passed': True, 'external_raw_images_read': 0})
    print('Intermediate spatial regions and actual-image verification complete', flush=True)


def main():
    torch.set_num_threads(4); torch.manual_seed(20261124)
    with (OUT / 'extraction.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run()
        except BaseException:
            dump(OUT / 'extraction_state.json', {'status': 'failed', 'pid': os.getpid(), 'error': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()
