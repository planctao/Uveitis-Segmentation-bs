"""Original M5 inputs plus development-only intermediate spatial regions."""
import json
import numpy as np
from experiment import KEYS


def load_derived(root, frame, digest, feature):
    assert feature in ['block4', 'block8']
    protocol = json.loads((root / 'runs/early_regions/protocol.json').read_text())
    indices = np.flatnonzero(frame.split.isin(['train', 'val']))
    assert np.array_equal(indices, np.arange(2091))
    with np.load(root / 'features/multiscale_vit.npz') as saved:
        assert str(saved['manifest_sha256']) == digest and np.array_equal(saved['path'], frame.path)
        values = {k: saved[k][indices].astype('float32') for k in KEYS}
    with np.load(protocol['feature_storage']) as saved:
        assert str(saved['manifest_sha256']) == digest and np.array_equal(saved['indices'], indices)
        assert np.array_equal(saved['path'], frame.iloc[indices].path)
        values['early_regions'] = saved['regions_' + feature].astype('float32')
    result = {}
    for key, value in values.items():
        complete = np.full((len(frame), *value.shape[1:]), np.nan, dtype='float32')
        complete[indices] = value
        assert np.isfinite(complete[:2091]).all() and np.isnan(complete[2091:]).all()
        result[key] = complete
    return result
