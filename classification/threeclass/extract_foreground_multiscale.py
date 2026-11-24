"""Development-only paired-scale fine regions, with resumable verified identities."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd
from PIL import Image
import timm
import torch
from timm.models.eva import checkpoint_filter_fn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from experiment import ROOT, dump
from extract_vit import WEIGHTS
from foreground_pool import foreground_fraction, region_pool

OUT = ROOT / 'runs/foreground_multiscale'


class DevelopmentImages(Dataset):
    def __init__(self, frame, indices):
        self.frame = frame;self.indices = indices
        self.transforms = [T.Compose([T.Resize((size, size), antialias=True), T.ToTensor(),
            T.Normalize((.485, .456, .406), (.229, .224, .225))]) for size in [320, 448]]
    def __len__(self):return len(self.indices)
    def __getitem__(self, i):
        index = int(self.indices[i]);row = self.frame.iloc[index]
        assert row.split in ['train', 'val']
        with Image.open(row.cache_path) as image:
            image = image.convert('RGB');a, b = [transform(image) for transform in self.transforms]
        return index, a, b


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


@torch.inference_mode()
def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert json.loads((ROOT / 'verification_foreground_pool.json').read_text())['passed']
    digest = hashlib.sha256((ROOT / 'metadata/manifest.csv').read_bytes()).hexdigest()
    assert digest == protocol['manifest_sha256']
    frame = pd.read_csv(ROOT / 'metadata/manifest.csv')
    indices = np.flatnonzero(frame.split.isin(['train', 'val']))
    assert len(indices) == 2091 and np.array_equal(indices, np.arange(2091))
    target = Path(protocol['feature_storage']);target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        saved = np.load(target)
        assert str(saved['manifest_sha256']) == digest and np.array_equal(saved['indices'], indices)
        assert (target.with_suffix('.json')).exists()
        return
    with np.load(ROOT / 'features/multiscale_vit.npz') as source:
        reference = {key: source[key] for key in ['global', 'regions', 'path', 'manifest_sha256']}
    assert str(reference['manifest_sha256']) == digest and np.array_equal(reference['path'], frame.path)
    model = timm.create_model('vit_base_patch16_dinov3_qkvb', pretrained=False, num_classes=0, img_size=320, dynamic_img_size=True)
    model.load_state_dict(checkpoint_filter_fn(torch.load(WEIGHTS, map_location='cpu', weights_only=True), model), strict=True)
    model.eval().cuda()
    shards = OUT / 'extraction_shards';shards.mkdir(parents=True, exist_ok=True)
    batch_size = 16;start = time.time();max_global_error = 0.;max_region_error = 0.
    for begin in range(0, len(indices), batch_size):
        chosen = indices[begin:begin+batch_size]
        shard = shards / f'{begin:05d}.npz'
        if shard.exists():
            saved = np.load(shard)
            assert np.array_equal(saved['indices'], chosen) and str(saved['manifest_sha256']) == digest
            max_global_error = max(max_global_error, float(saved['global_max_error']))
            max_region_error = max(max_region_error, float(saved['region_max_error']))
            continue
        loader = DataLoader(DevelopmentImages(frame, chosen), batch_size=batch_size, num_workers=2, pin_memory=True)
        actual_indices, a, b = next(iter(loader))
        assert np.array_equal(actual_indices.numpy(), chosen)
        values = {'indices': chosen, 'manifest_sha256': np.array(digest)}
        global_errors, region_errors = [], []
        for view, (size, images) in enumerate([(320, a), (448, b)]):
            images = images.cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16):
                tokens = model.forward_features(images)
            patch_grid = size // 16
            patches = tokens[:, model.num_prefix_tokens:].transpose(1, 2).reshape(len(chosen), 768, patch_grid, patch_grid)
            occupancy = foreground_fraction(images, patch_grid)
            expected = reference['global'][chosen, view*768:(view+1)*768].astype('float32')
            global_errors.append(float(np.abs(tokens[:, 0].float().cpu().numpy() - expected).max()))
            for grid in protocol['grids']:
                ordinary, weighted, fractions = region_pool(patches, occupancy, grid)
                values[f'ordinary_{grid}_{size}'] = ordinary.cpu().numpy().astype('float16')
                values[f'weighted_{grid}_{size}'] = weighted.cpu().numpy().astype('float16')
                values[f'occupancy_{grid}_{size}'] = fractions.cpu().numpy().astype('float32')
                if grid == 3:
                    expected_regions = reference['regions'][chosen, :, view*768:(view+1)*768].astype('float32')
                    region_errors.append(float(np.abs(ordinary.cpu().numpy() - expected_regions).max()))
        values['global_max_error'] = np.array(max(global_errors));values['region_max_error'] = np.array(max(region_errors))
        assert values['global_max_error'] < .03 and values['region_max_error'] < .03
        temporary = shard.with_suffix('.tmp')
        with temporary.open('wb') as stream:np.savez_compressed(stream, **values)
        temporary.replace(shard)
        max_global_error = max(max_global_error, float(values['global_max_error']))
        max_region_error = max(max_region_error, float(values['region_max_error']))
        state = {'status': 'extracting_development_regions', 'pid': os.getpid(), 'completed_images': begin+len(chosen),
            'total_images': len(indices), 'seconds': time.time()-start, 'max_reference_global_error': max_global_error,
            'max_reference_coarse_region_error': max_region_error, 'external_raw_images_read': 0}
        dump(OUT / 'extraction_state.json', state)
        if begin % (batch_size*10) == 0:print(json.dumps(state), flush=True)
    banks = {f'{pool}_{grid}': [] for pool in ['ordinary', 'weighted'] for grid in protocol['grids']}
    banks.update({f'occupancy_{grid}': [] for grid in protocol['grids']})
    for begin in range(0, len(indices), batch_size):
        saved = np.load(shards / f'{begin:05d}.npz')
        assert np.array_equal(saved['indices'], indices[begin:begin+batch_size])
        for grid in protocol['grids']:
            for pool in ['ordinary', 'weighted']:
                banks[f'{pool}_{grid}'].append(np.concatenate([saved[f'{pool}_{grid}_{size}'] for size in [320, 448]], -1))
            banks[f'occupancy_{grid}'].append(.5 * (saved[f'occupancy_{grid}_320'] + saved[f'occupancy_{grid}_448']))
    arrays = {key: np.concatenate(values) for key, values in banks.items()}
    arrays.update(indices=indices, path=frame.iloc[indices].path.to_numpy(dtype=str), manifest_sha256=np.array(digest))
    temporary = target.with_suffix('.tmp')
    with temporary.open('wb') as stream:np.savez_compressed(stream, **arrays)
    temporary.replace(target)
    weights_hash = hashlib.sha256(Path(WEIGHTS).read_bytes()).hexdigest()
    metadata = {'manifest_sha256': digest, 'protocol_sha256': hashlib.sha256((OUT / 'protocol.json').read_bytes()).hexdigest(),
        'weights_sha256': weights_hash, 'weights': WEIGHTS, 'strict_loading': True, 'development_images': len(indices),
        'external_raw_images_read': 0, 'seconds': time.time()-start,
        'max_reference_global_error': max_global_error, 'max_reference_coarse_region_error': max_region_error,
        'arrays': {key: {'shape': list(value.shape), 'dtype': str(value.dtype), 'value_sha256': array_hash(value)} for key, value in arrays.items()}}
    dump(target.with_suffix('.json'), metadata)
    dump(OUT / 'extraction_state.json', {'status': 'completed', 'pid': os.getpid(), 'development_images': len(indices),
        'bank': str(target), 'external_raw_images_read': 0})
    print(json.dumps(metadata), flush=True)


def main():
    torch.set_num_threads(4);torch.manual_seed(20261124)
    with (OUT / 'extraction.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (OUT / 'extraction.pid').write_text(str(os.getpid()))
        try:run()
        except BaseException:
            dump(OUT / 'extraction_state.json', {'status': 'failed', 'pid': os.getpid(), 'error': traceback.format_exc()})
            raise


if __name__ == '__main__':main()
