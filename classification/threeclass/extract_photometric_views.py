"""Extract and audit fixed luminance views; never open an external image."""
import fcntl
import json
import os
from pathlib import Path
import time
import traceback
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
from timm.models.eva import checkpoint_filter_fn
from experiment import ROOT, KEYS, dump
from extract_vit import WEIGHTS
from kernel_residual_experiments import sha
from photometric_data import PhotometricImages, luminance_view, load_derived

OUT = ROOT / 'runs/photometric_views'


def array_hash(value):
    import hashlib
    x = np.ascontiguousarray(value)
    h = hashlib.sha256(); h.update(str(x.dtype).encode()); h.update(str(x.shape).encode()); h.update(x.tobytes())
    return h.hexdigest()


def backbone():
    model = timm.create_model('vit_base_patch16_dinov3_qkvb', pretrained=False,
                             num_classes=0, img_size=320, dynamic_img_size=True)
    state = torch.load(WEIGHTS, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint_filter_fn(state, model), strict=True)
    return model.eval().cuda()


@torch.inference_mode()
def features_for(model, a, b):
    captured = {}
    hook = model.blocks[7].register_forward_hook(lambda module, inputs, output: captured.update(mid=output))
    views = []
    try:
        for size, images in [(320, a), (448, b)]:
            with torch.autocast('cuda', dtype=torch.float16):
                tokens = model.forward_features(images.cuda(non_blocking=True))
                patch = tokens[:, model.num_prefix_tokens:]
                regions = F.adaptive_avg_pool2d(patch.transpose(1, 2).reshape(len(images), 768,
                    size // 16, size // 16), (3, 3)).flatten(2).transpose(1, 2)
                values = {'global': tokens[:, 0], 'regions': regions,
                          'mid': captured.pop('mid')[:, model.num_prefix_tokens:].mean(1),
                          'dispersion': patch.float().std(1)}
            views.append({k: v.cpu().numpy().astype('float16') for k, v in values.items()})
    finally:
        hook.remove()
    return {k: np.concatenate([v[k] for v in views], -1) for k in KEYS}


def original_control(model, frame):
    ids = np.array([0, 352, 887, 1501, 1800, 2090])
    loader = DataLoader(PhotometricImages(frame, ids, corrected=False), batch_size=3, num_workers=0)
    collected = {k: [] for k in KEYS}
    for actual, a, b in loader:
        values = features_for(model, a, b)
        for k in KEYS:
            collected[k].append(values[k])
    differences = {}
    with np.load(ROOT / 'features/multiscale_vit.npz') as saved:
        for key in KEYS:
            actual = np.concatenate(collected[key]).astype('float32'); expected = saved[key][ids].astype('float32')
            relative = float(np.abs(actual - expected).mean() / (np.abs(expected).mean() + 1e-7))
            assert relative < .003, (key, relative)
            differences[key] = {'relative_mae': relative, 'max_abs_error': float(np.abs(actual - expected).max())}
    dump(OUT / 'original_input_control.json', {'passed': True, 'development_indices': ids.tolist(),
         'unchanged_pipeline_feature_differences': differences})


def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    digest = sha(ROOT / 'metadata/manifest.csv'); assert digest == protocol['manifest_sha256']
    assert sha(Path(WEIGHTS)) == protocol['pretrained_weights']['sha256']
    frame = pd.read_csv(ROOT / 'metadata/manifest.csv')
    indices = np.flatnonzero(frame.split.isin(['train', 'val']))
    assert np.array_equal(indices, np.arange(2091))
    cv2.setNumThreads(0)
    model = backbone(); original_control(model, frame)
    target = Path(protocol['feature_storage']); target.parent.mkdir(parents=True, exist_ok=True)
    shards = OUT / 'extraction_shards'; shards.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(PhotometricImages(frame, indices), batch_size=16, num_workers=4, pin_memory=True)
    begin_time = time.time()
    for step, (actual, a, b) in enumerate(loader):
        chosen = indices[step * 16:(step + 1) * 16]; assert np.array_equal(actual.numpy(), chosen)
        shard = shards / f'{step:05d}.npz'
        if shard.exists():
            with np.load(shard) as saved:
                assert np.array_equal(saved['indices'], chosen) and str(saved['manifest_sha256']) == digest
                assert all(np.isfinite(saved[k]).all() for k in KEYS)
        else:
            values = features_for(model, a, b)
            with shard.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream, indices=chosen, manifest_sha256=np.array(digest), **values)
            shard.with_suffix('.tmp').replace(shard)
        if step % 10 == 0 or step + 1 == len(loader):
            state = {'status': 'extracting', 'pid': os.getpid(), 'completed_images': int(chosen[-1]) + 1,
                     'development_images': 2091, 'seconds': time.time() - begin_time, 'external_raw_images_read': 0}
            dump(OUT / 'extraction_state.json', state); print(json.dumps(state), flush=True)
    arrays = {k: [] for k in KEYS}
    for step in range(len(loader)):
        with np.load(shards / f'{step:05d}.npz') as saved:
            assert np.array_equal(saved['indices'], indices[step * 16:(step + 1) * 16])
            for k in KEYS:
                arrays[k].append(saved[k])
    arrays = {k: np.concatenate(v) for k, v in arrays.items()}
    arrays.update(indices=indices, path=frame.iloc[indices].path.to_numpy(dtype=str), manifest_sha256=np.array(digest))
    if target.exists():
        with np.load(target) as saved:
            assert set(saved.files) == set(arrays)
            assert all(np.array_equal(saved[k], v) for k, v in arrays.items())
    else:
        with target.with_suffix('.tmp').open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        target.with_suffix('.tmp').replace(target)
    dump(target.with_suffix('.json'), {'manifest_sha256': digest, 'weights_sha256': protocol['pretrained_weights']['sha256'],
         'feature_sha256': sha(target), 'opencv': cv2.__version__, 'strict_loading': True,
         'development_images': 2091, 'external_raw_images_read': 0,
         'arrays': {k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'sha256': array_hash(v)} for k, v in arrays.items()}})
    dump(OUT / 'extraction_state.json', {'status': 'verifying', 'pid': os.getpid(), 'external_raw_images_read': 0})
    # All shards independently checked against the completed bank, not just against their identity fields.
    checked = 0
    for step in range(len(loader)):
        with np.load(shards / f'{step:05d}.npz') as saved:
            rows = saved['indices']
            for k in KEYS:
                assert np.array_equal(saved[k], arrays[k][rows])
            checked += 1
    del model; torch.cuda.empty_cache()
    model = backbone(); selected = np.array([0, 352, 887, 1501, 1800, 2090]); reconstructions = []
    for index, a, b in DataLoader(PhotometricImages(frame, selected), batch_size=1, num_workers=0):
        actual = features_for(model, a, b); row = int(index[0]); differences = {}
        for k in KEYS:
            expected = arrays[k][[row]].astype('float32'); value = actual[k].astype('float32')
            relative = float(np.abs(value - expected).mean() / (np.abs(expected).mean() + 1e-7))
            assert relative < .003, (row, k, relative)
            differences[k] = {'relative_mae': relative, 'max_abs_error': float(np.abs(value - expected).max())}
        with Image.open(frame.iloc[row].cache_path) as image:
            rgb = np.asarray(image.convert('RGB')).copy()
        view = luminance_view(rgb); background = rgb.max(2) <= 10
        assert np.array_equal(view[background], rgb[background])
        assert np.any(view[~background] != rgb[~background])
        reconstructions.append({'development_index': row, 'feature_differences': differences,
                                'background_pixels_unchanged': int(background.sum())})
    # The operation has no dataset-fit state and never changes a source file.
    black = np.zeros((448, 448, 3), dtype='uint8'); assert np.array_equal(luminance_view(black), black)
    for feature in ['clahe_dual', 'original_clahe']:
        values = load_derived(ROOT, frame, digest, feature)
        assert all(np.isfinite(v[:2091]).all() and np.isnan(v[2091:]).all() for v in values.values())
    dump(ROOT / 'verification_photometric_views.json', {'passed': True, 'manifest_sha256': digest,
         'weights_sha256': protocol['pretrained_weights']['sha256'], 'development_images': 2091,
         'shards_checked': checked, 'actual_images_recomputed': reconstructions,
         'external_raw_images_read': 0, 'checks': [
             'Pinned B16 weight bytes verified before strict loading; original six-image pipeline control passed',
             'All CLAHE development feature shards exactly match final bank and row identities',
             'Freshly reloaded backbone reproduced six actual corrected images at both resolutions',
             'Low-intensity background pixels untouched; all-black boundary case preserved',
             'Both predeclared layouts reconstructed; all external rows explicitly NaN']})
    dump(OUT / 'extraction_state.json', {'status': 'completed', 'pid': os.getpid(),
         'development_images': 2091, 'verification_passed': True, 'external_raw_images_read': 0})
    print('Photometric development feature bank and actual-image verification complete', flush=True)


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
