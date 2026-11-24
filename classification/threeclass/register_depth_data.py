"""Load verified three-depth register streams without external feature rows."""
import json
import numpy as np
import pandas as pd
from experiment import ROOT
from kernel_residual_experiments import sha
from dinov3_large_data import load_derived
from register_depth_features import TOKEN_KEYS


def load_all():
    frame = pd.read_csv(ROOT/'metadata/manifest.csv'); digest = sha(ROOT/'metadata/manifest.csv')
    proof = json.loads((ROOT/'verification_register_depth_bank.json').read_text())
    assert proof['passed'] and proof['manifest_sha256'] == digest
    assert proof['classification_fits'] == proof['classification_scores'] == 0 and not proof['external_test_evaluated']
    assert sha(proof['feature_storage']) == proof['feature_bank_sha256']
    features = load_derived(ROOT, frame, digest, 'large_dual')
    with np.load(proof['feature_storage']) as bank:
        assert np.array_equal(bank['indices'], np.arange(2091)) and str(bank['manifest_sha256']) == digest
        assert np.array_equal(bank['path'], frame.iloc[:2091].path.to_numpy(dtype=str))
        for key in TOKEN_KEYS:
            assert bank[key].shape == (2091, 4, 2048)
            value = np.full((len(frame), 4, 2048), np.nan, dtype='float32'); value[:2091] = bank[key]
            assert np.isfinite(value[:2091]).all(); features[key] = value
    return frame, features, digest
