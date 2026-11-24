"""Load only verified development B token evidence; external rows stay NaN."""
import json
import numpy as np
import pandas as pd
from experiment import ROOT
from kernel_residual_experiments import sha
from experiment import load_data
from base_token_features import TOKEN_KEYS


def load_all():
    frame = pd.read_csv(ROOT/'metadata/manifest.csv'); digest = sha(ROOT/'metadata/manifest.csv')
    verification = json.loads((ROOT/'verification_base_token_feature_bank.json').read_text())
    assert verification['passed'] and verification['manifest_sha256'] == digest
    assert verification['classification_fits'] == verification['classification_scores'] == 0 and not verification['external_test_evaluated']
    path = verification['feature_storage']; assert sha(path) == verification['feature_bank_sha256']
    loaded_frame, base, loaded_digest = load_data('multiscale_vit'); tokens = {}
    assert loaded_frame.equals(frame) and loaded_digest == digest
    for value in base.values(): value[2091:] = np.nan
    with np.load(path) as bank:
        assert np.array_equal(bank['indices'], np.arange(2091)) and str(bank['manifest_sha256']) == digest
        assert np.array_equal(bank['path'], frame.iloc[:2091].path.to_numpy(dtype=str))
        for key in TOKEN_KEYS:
            count = verification['token_counts'][key]; assert bank[key].shape == (2091, count, 1536)
            tokens[key] = np.full((len(frame), count, 1536), np.nan, dtype='float32'); tokens[key][:2091] = bank[key]
    return frame, base, tokens, digest
