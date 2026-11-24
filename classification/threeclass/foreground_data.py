"""Paired-scale foreground regions for development rows; global paths unchanged."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment import ROOT, load_data
from foreground_pool import valid_regions

OUT = ROOT / 'runs/foreground_multiscale'


def load_foreground(arm):
    frame, original, digest = load_data('multiscale_vit')
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert digest == protocol['manifest_sha256']
    identity = next(item for item in protocol['configuration_plan'] if item['name'] == arm)
    with np.load(Path(protocol['feature_storage'])) as bank:
        indices = bank['indices']
        assert np.array_equal(indices, np.flatnonzero(frame.split.isin(['train', 'val'])))
        assert np.array_equal(bank['path'], frame.iloc[indices].path)
        assert str(bank['manifest_sha256']) == digest
        local = bank[f'{identity["pool"]}_{identity["grid"]}'].astype('float32')
        occupancy = bank[f'occupancy_{identity["grid"]}']
    # Every row outside development has missing regional evidence, and this loader
    # is only accepted by the dedicated development runner which audits all masks.
    # NaNs make accidental external scoring fail instead of silently supplying zeros.
    regions = np.full((len(frame), local.shape[1], local.shape[2]), np.nan, dtype='float32')
    regions[indices] = local
    valid = valid_regions(torch.from_numpy(occupancy)).numpy().astype('float32')
    mask = np.full((len(frame), local.shape[1]), np.nan, dtype='float32');mask[indices] = valid
    features = {key: value for key, value in original.items() if key != 'regions'}
    features.update(regions=regions)
    if identity['use_region_mask']:features['region_mask'] = mask
    return frame, features, digest
