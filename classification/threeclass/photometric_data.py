"""Fixed luminance-only CLAHE views, development rows and feature layouts."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from experiment import KEYS


def luminance_view(rgb):
    rgb = np.asarray(rgb)
    assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[-1] == 3
    mask = rgb.max(2) > 10
    if not mask.any():
        return rgb.copy()
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8)).apply(lab[:, :, 0])
    corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    corrected[~mask] = rgb[~mask]
    return corrected


class PhotometricImages(Dataset):
    def __init__(self, frame, indices, corrected=True):
        self.frame = frame; self.indices = indices; self.corrected = corrected
        self.transforms = [T.Compose([T.Resize((size, size), antialias=True), T.ToTensor(),
            T.Normalize((.485, .456, .406), (.229, .224, .225))]) for size in [320, 448]]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        index = int(self.indices[i]); row = self.frame.iloc[index]
        assert row.split in ['train', 'val']
        with Image.open(row.cache_path) as image:
            rgb = np.asarray(image.convert('RGB')).copy()
        if self.corrected:
            rgb = luminance_view(rgb)
        image = Image.fromarray(rgb)
        a, b = [transform(image) for transform in self.transforms]
        return index, a, b


def load_derived(root, frame, digest, feature):
    assert feature in ['clahe_dual', 'original_clahe']
    protocol = json.loads((root / 'runs/photometric_views/protocol.json').read_text())
    with np.load(protocol['feature_storage']) as saved:
        indices = np.flatnonzero(frame.split.isin(['train', 'val']))
        assert np.array_equal(indices, np.arange(2091)) and np.array_equal(saved['indices'], indices)
        assert np.array_equal(saved['path'], frame.iloc[indices].path)
        assert str(saved['manifest_sha256']) == digest
        values = {k: saved[k].astype('float32') for k in KEYS}
    if feature == 'original_clahe':
        with np.load(root / 'features/multiscale_vit.npz') as original:
            assert str(original['manifest_sha256']) == digest and np.array_equal(original['path'], frame.path)
            values = {k: np.concatenate([original[k][indices].astype('float32'), values[k]], -1) for k in KEYS}
    result = {}
    for key, x in values.items():
        complete = np.full((len(frame), *x.shape[1:]), np.nan, dtype='float32'); complete[indices] = x
        assert np.isfinite(complete[:2091]).all() and np.isnan(complete[2091:]).all()
        result[key] = complete
    return result
