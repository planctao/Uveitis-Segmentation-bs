"""Deterministic reflected inputs and lazily derived paired-view feature banks."""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

KEYS = ['global', 'regions', 'mid', 'dispersion']
PERMUTATION = np.array([2, 1, 0, 5, 4, 3, 8, 7, 6])
DERIVED = {'multiscale_mirror_mean', 'multiscale_mirror_bag', 'multiscale_mirror_pairs'}


class MirrorImages(Dataset):
    def __init__(self, frame, mirror=True):
        self.paths = frame.cache_path.tolist()
        self.mirror = mirror
        self.transforms = [T.Compose([T.Resize((size, size), antialias=True), T.ToTensor(),
            T.Normalize((.485, .456, .406), (.229, .224, .225))]) for size in [320, 448]]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as source:
            im = source.convert('RGB')
            if self.mirror:
                im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            a, b = [transform(im) for transform in self.transforms]
        return a, b, index


def source_arrays(root, frame, digest):
    banks = []
    for name in ['multiscale_vit', 'multiscale_flip']:
        with np.load(root / 'features' / f'{name}.npz') as saved:
            assert str(saved['manifest_sha256']) == digest
            assert np.array_equal(saved['path'], frame.path.to_numpy(dtype=str))
            banks.append({key: saved[key].astype('float32') for key in KEYS})
    # Put the mirror's regional evidence back into the original image coordinates.
    banks[1]['regions'] = banks[1]['regions'][:, PERMUTATION]
    return banks


def load_derived(root, frame, digest, feature):
    assert feature in DERIVED
    original, reflected = source_arrays(root, frame, digest)
    if feature == 'multiscale_mirror_pairs':
        # Feature normalization pools BOTH views, tying their coordinate systems.
        return {key: np.stack([original[key], reflected[key]], axis=1) for key in KEYS}
    values = {key: ((original[key] + reflected[key]) * .5).astype('float16').astype('float32')
              for key in KEYS}
    if feature == 'multiscale_mirror_bag':
        values['regions'] = np.concatenate([original['regions'], reflected['regions']], axis=1)
    return values
