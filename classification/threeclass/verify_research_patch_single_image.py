"""Supplement fixed-batch patch inference with actual one-image requests."""
import fcntl
import json
import os
import time
import traceback
import numpy as np
import pandas as pd
import torch
from experiment import ROOT, KEYS, dump
from kernel_residual_experiments import sha
from large_token_data import load_all
from research_patch_image_inference import (
    FrozenPatchImageExtractor, restore_reviewed_adapter, adapter_probability)
from register_patch_query_features import BANK_KEYS

OUT = ROOT/'runs/research_patch_single_image_check'
DEST = ROOT/'results/research_patch_single_image_check.json'


def check_frozen(protocol):
    for section in ['code_sha256', 'frozen_artifacts', 'image_sha256']:
        for name, expected in protocol[section].items():
            assert sha(name) == expected, name


def main_run():
    started = time.time()
    assert not DEST.exists()
    protocol = json.loads((OUT/'protocol.json').read_text())
    check_frozen(protocol)
    dump(OUT/'state.json', {'status': 'loading', 'pid': os.getpid(), 'external_test_evaluated': False})
    frame, features, tokens, digest = load_all()
    assert digest == protocol['manifest_sha256']
    bank = {name: np.load(ROOT/'runs/register_patch_query_bank/arrays'/(name+'.npy'), mmap_mode='r')
            for name in BANK_KEYS}
    extractor = FrozenPatchImageExtractor()
    live_cases = []
    for index in protocol['validation_indices']:
        assert frame.iloc[index].split == 'val'
        dump(OUT/'state.json', {'status': 'checking_single_image', 'pid': os.getpid(),
            'manifest_index': index, 'external_test_evaluated': False})
        image = extractor.prepare_images([frame.iloc[index].path])
        _, live_large, live_bank = extractor.extract_patch_preprocessed(image)
        expected_large = {name: features[name][index:index+1] for name in KEYS}
        expected_large['side_tokens'] = tokens['registers'][index:index+1]
        expected_bank = {name: np.asarray(value[index:index+1]) for name, value in bank.items()}
        overlaps = []
        for view in range(2):
            for head in range(16):
                a = set(live_bank['patch_indices'][0, view, head].tolist())
                b = set(expected_bank['patch_indices'][0, view, head].tolist())
                overlaps.append(len(a & b)/64)
        errors = {}
        for name in live_large:
            assert np.allclose(live_large[name], expected_large[name], atol=.05, rtol=.005), name
            errors[name] = float(np.abs(live_large[name]-expected_large[name]).max())
        live_cases.append({'index': index, 'large': live_large, 'bank': live_bank,
            'cache_large': expected_large, 'cache_bank': expected_bank,
            'head_patch_overlap_fraction': overlaps, 'large_feature_errors': errors})
    rows = []
    for arm in protocol['adapter_arms']:
        model, source = restore_reviewed_adapter('register_patch_joint', arm, 'validation', 1)
        for case in live_cases:
            live_p = adapter_probability(model, source, case['large'], case['bank'])
            cache_p = adapter_probability(model, source, case['cache_large'], case['cache_bank'])
            maximum = float(np.abs(live_p-cache_p).max())
            assert np.allclose(live_p, cache_p, atol=.005, rtol=.01), (arm, case['index'], maximum)
            assert np.array_equal(live_p.argmax(1), cache_p.argmax(1)), (arm, case['index'])
            rows.append({'arm': arm, 'validation_index': case['index'],
                'maximum_single_image_vs_cached_probability_error': maximum,
                'classification_decision_matches_cached': True})
        del model
        torch.cuda.empty_cache()
    check_frozen(protocol)
    cases = [{
        'validation_index': case['index'],
        'minimum_head_patch_overlap_fraction': min(case['head_patch_overlap_fraction']),
        'mean_head_patch_overlap_fraction': float(np.mean(case['head_patch_overlap_fraction'])),
        'all32head_patch_overlap_fractions': case['head_patch_overlap_fraction'],
        'large_feature_maximum_errors': case['large_feature_errors'],
    } for case in live_cases]
    dump(DEST, {'passed': True, 'protocol_sha256': sha(OUT/'protocol.json'),
        'manifest_sha256': digest, 'single_original_validation_images_checked': len(cases),
        'reviewed_adapter_models_checked': len(protocol['adapter_arms']),
        'adapter_image_checks': len(rows), 'image_cases': cases, 'adapter_checks': rows,
        'maximum_probability_error': max(row['maximum_single_image_vs_cached_probability_error'] for row in rows),
        'feature_tolerance': {'atol': .05, 'rtol': .005},
        'single_image_probability_tolerance': {'atol': .005, 'rtol': .01},
        'scope_limit': 'Engineering robustness check on exactly3preselectedvalidationimages and4completedjointadaptercheckpoints, not an accuracy score or gain test. Probability tolerance is preregistered separately from the exact-batch check because one-image float16 kernels can select different near-tied top64 positions. No exact bank equivalence is claimed for different batch shapes. Other adapters and full validation inference need their own candidate-specific checks.',
        'new_model_fits': 0, 'new_candidate_scores': 0, 'external_images_read': 0,
        'external_test_evaluated': False, 'seconds': time.time()-started,
        'code_sha256': protocol['code_sha256']})
    dump(OUT/'state.json', {'status': 'completed', 'pid': os.getpid(), 'passed': True,
        'external_test_evaluated': False})
    print(json.dumps({'passed': True, 'images': len(cases), 'adapter_image_checks': len(rows),
        'maximum_probability_error': max(row['maximum_single_image_vs_cached_probability_error'] for row in rows),
        'minimum_patch_overlap': min(row['minimum_head_patch_overlap_fraction'] for row in cases)}), flush=True)


def main():
    torch.set_num_threads(4)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            main_run()
        except BaseException:
            dump(OUT/'state.json', {'status': 'failed', 'pid': os.getpid(), 'error': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()
