"""Verified full-set attention mass and partition-local control statistics."""
import json
import numpy as np
from experiment import ROOT
from kernel_residual_experiments import sha
from register_patch_query_data import load_all as original_load
from register_patch_query_features import BANK_KEYS

MASS_OUT = ROOT/'runs/register_patch_attention_mass_bank_v2'
MASS_KEY = 'full_selected_mass'
INPUT_KEYS = BANK_KEYS+[MASS_KEY]


def fit_mass_statistics(bank, fit):
    fit = np.asarray(fit, dtype='int64')
    assert fit.ndim == 1 and len(fit) and len(np.unique(fit)) == len(fit)
    assert fit.min() >= 0 and fit.max() < 2091
    mass = np.asarray(bank[MASS_KEY][fit], dtype='float64')
    assert mass.shape == (len(fit), 2, 16, 4) and np.isfinite(mass).all()
    assert (mass > 0).all() and (mass <= 1).all()
    return mass.mean(axis=(0, 2, 3))


def load_all():
    review_path = ROOT/'results/register_patch_attention_mass_bank_v2_root_review.json'
    assert sha(review_path) == 'aa7239b36390333760db26e524ad5716bd46403dc1a5e95e67fea0251e2d13d6'
    review = json.loads(review_path.read_text())
    assert review['passed'] and review['chunks_checked'] == 262 and len(review['saved_cpu_witnesses']) == 6
    state = json.loads((MASS_OUT/'state.json').read_text())
    assert state['status'] == 'completed', 'Never use a failed or active mass cache.'
    proof = json.loads((MASS_OUT/'verification.json').read_text())
    assert sha(MASS_OUT/'verification.json') == '5e41cd49edf5d8e8c8de7dba6a93dc02722afef3d3e9d88a96d74704cdb4e807'
    assert proof['passed'] and proof['development_images'] == 2091
    assert proof['classification_fits'] == proof['classification_scores'] == proof['classifier_probability_predictions'] == proof['external_images_read'] == 0
    assert proof['backbone_view_forwards'] == 530
    assert proof['all_six_original_bank_arrays_reproduced_bitwise_for_every_image']
    assert sha(MASS_OUT/'protocol.json') == proof['protocol_sha256']
    protocol = json.loads((MASS_OUT/'protocol.json').read_text())
    for name, expected in protocol['code_sha256'].items():
        assert sha(name) == expected, name
    for name, expected in proof['frozen_artifacts'].items():
        assert sha(name) == expected, name
    frame, features, bank, tail, digest = original_load()
    assert digest == proof['manifest_sha256']
    mass = np.load(MASS_OUT/'selected_mass.npy', mmap_mode='r')
    assert mass.shape == (2091, 2, 16, 4) and mass.dtype == np.dtype('float64')
    assert np.isfinite(mass).all() and (mass > 0).all() and (mass <= 1).all()
    return frame, features, {**bank, MASS_KEY: mass}, tail, digest
