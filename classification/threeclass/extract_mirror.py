"""Extract frozen 320+448 reflected-image features, with indexed resume shards."""
import hashlib
import json
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm
from timm.models.eva import checkpoint_filter_fn
from torch.utils.data import DataLoader
from experiment import ROOT, dump
from extract_vit import WEIGHTS
from mirror_data import MirrorImages, KEYS


def sha(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(20261124)
    frame = pd.read_csv(ROOT / 'metadata/manifest.csv')
    digest = sha(ROOT / 'metadata/manifest.csv')
    destination = ROOT / 'features/multiscale_flip.npz'
    if destination.exists():
        with np.load(destination) as saved:
            assert str(saved['manifest_sha256']) == digest
            assert np.array_equal(saved['path'], frame.path.to_numpy(dtype=str))
        print('Existing reflected features match manifest', flush=True)
        return
    model = timm.create_model('vit_base_patch16_dinov3_qkvb', pretrained=False,
                              num_classes=0, img_size=320, dynamic_img_size=True)
    state = torch.load(WEIGHTS, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint_filter_fn(state, model), strict=True)
    del state
    model.eval().cuda()
    captured = {}
    handle = model.blocks[7].register_forward_hook(lambda module, inputs, output: captured.update(mid=output))

    def extract(a, b):
        views = []
        for size, x in [(320, a), (448, b)]:
            with torch.autocast('cuda', dtype=torch.float16):
                tokens = model.forward_features(x.cuda(non_blocking=True))
                patches = tokens[:, model.num_prefix_tokens:]
                regions = F.adaptive_avg_pool2d(patches.transpose(1, 2).reshape(len(x), 768,
                    size // 16, size // 16), (3, 3)).flatten(2).transpose(1, 2)
                values = {'global': tokens[:, 0], 'regions': regions,
                    'mid': captured.pop('mid')[:, model.num_prefix_tokens:].mean(1),
                    'dispersion': patches.float().std(1)}
            views.append({key: value.cpu().numpy().astype('float16') for key, value in values.items()})
        return {key: np.concatenate([view[key] for view in views], axis=-1) for key in KEYS}

    sample = frame[frame.split.eq('train')].iloc[:8]
    sample_loader = DataLoader(MirrorImages(sample, mirror=False), batch_size=4)
    calculated = {key: [] for key in KEYS}
    for a, b, _ in sample_loader:
        values = extract(a, b)
        for key in KEYS:
            calculated[key].append(values[key])
    differences = {}
    with np.load(ROOT / 'features/multiscale_vit.npz') as saved:
        for key in KEYS:
            actual = np.concatenate(calculated[key]).astype('float32')
            expected = saved[key][sample.index].astype('float32')
            relative = float(np.abs(actual - expected).mean() / (np.abs(expected).mean() + 1e-7))
            differences[key] = {'relative_mean_absolute_error': relative,
                                'max_absolute_error': float(np.abs(actual - expected).max())}
            assert relative < .003, (key, differences[key])
    dump(ROOT / 'runs/mirror_study/input_verification.json', {'passed': True,
         'original_images_checked': len(sample), 'unmodified_pipeline_cache_differences': differences,
         'policy': 'Same decoded 448 JPEG, Resize, ImageNet normalization, dynamic DINO weights and float16 feature representation; reflection is the only input intervention.'})
    shard_dir = ROOT / 'feature_shards/multiscale_flip'
    shard_dir.mkdir(parents=True, exist_ok=True)
    protocol = {'manifest_sha256': digest, 'weights_sha256': sha(WEIGHTS), 'sizes': [320, 448],
        'batch': 4, 'timm': timm.__version__, 'torch': torch.__version__,
        'transformation': 'Horizontal reflection of decoded existing 448 JPEG, before each input Resize.'}
    path = shard_dir / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol
    else:
        dump(path, protocol)
    loader = DataLoader(MirrorImages(frame), batch_size=4, num_workers=4, pin_memory=True)
    outputs = {key: [] for key in KEYS}
    started = time.time()
    for step, (a, b, indices) in enumerate(loader):
        shard = shard_dir / f'{step:05d}.npz'
        if shard.exists():
            with np.load(shard) as saved:
                assert np.array_equal(saved['indices'], indices.numpy())
                values = {key: saved[key] for key in KEYS}
        else:
            values = extract(a, b)
            with shard.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream, indices=indices.numpy(), **values)
            os.replace(shard.with_suffix('.tmp'), shard)
        for key in KEYS:
            assert np.isfinite(values[key]).all() and len(values[key]) == len(indices)
            outputs[key].append(values[key])
        if step % 25 == 0 or step + 1 == len(loader):
            progress = {'status': 'extracting', 'pid': os.getpid(), 'batch': step + 1,
                'total_batches': len(loader), 'processed_images': int(indices[-1]) + 1,
                'total_images': len(frame), 'seconds': time.time() - started,
                'external_metrics_evaluated': False}
            dump(ROOT / 'runs/mirror_study/state.json', progress)
            print(json.dumps(progress), flush=True)
    handle.remove()
    arrays = {key: np.concatenate(values) for key, values in outputs.items()}
    arrays.update(path=frame.path.to_numpy(dtype=str), manifest_sha256=np.array(digest))
    with destination.with_suffix('.tmp').open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(destination.with_suffix('.tmp'), destination)
    dump(destination.with_suffix('.json'), {**protocol, 'strict_loading': True,
         'images': len(frame), 'seconds': time.time() - started,
         'feature_sha256': sha(destination), 'shapes': {k: list(v.shape) for k, v in arrays.items()}})
    print('Reflected dual-scale features completed', flush=True)


if __name__ == '__main__':
    main()
