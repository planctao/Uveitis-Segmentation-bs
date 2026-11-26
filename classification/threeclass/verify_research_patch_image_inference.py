"""Check raw-image patch features and reviewed adapters without classification scores."""
import fcntl
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd
import torch
from experiment import ROOT, KEYS, load_data, dump
from kernel_residual_experiments import sha
from large_token_data import load_all
from research_patch_image_inference import (
    FrozenPatchImageExtractor, restore_reviewed_adapter, adapter_probability)
from register_patch_query_features import BANK_KEYS

OUT = ROOT/'runs/research_patch_image_inference_check'
DEST = ROOT/'results/research_patch_image_inference_check.json'


def main_run():
    started = time.time()
    assert not DEST.exists()
    protocol = json.loads((OUT/'protocol.json').read_text())
    for section in ['code_sha256', 'frozen_artifacts', 'image_sha256']:
        for name, expected in protocol[section].items():
            assert sha(name) == expected, name
    dump(OUT/'state.json', {'status': 'loading', 'pid': os.getpid(),
        'external_test_evaluated': False})
    frame, large, tokens, digest = load_all()
    _, base, other_digest = load_data('multiscale_vit')
    assert digest == other_digest == protocol['manifest_sha256']
    expected_bank = {name: np.load(ROOT/'runs/register_patch_query_bank/arrays'/(name+'.npy'),
        mmap_mode='r') for name in BANK_KEYS}
    extractor = FrozenPatchImageExtractor()
    cases = []
    checkpoint_rows = json.loads((ROOT/'results/research_register_patch_joint/verification.json').read_text())['fits']
    arms = protocol['adapter_arms']
    for selected in protocol['development_batches']:
        indices = np.asarray(selected, dtype='int64')
        assert len(indices) == 8 and indices[0] % 8 == 0
        assert np.array_equal(indices, np.arange(indices[0], indices[0]+8))
        assert frame.iloc[indices].split.isin(['train', 'val']).all()
        dump(OUT/'state.json', {'status': 'checking_images', 'pid': os.getpid(),
            'manifest_indices': indices.tolist(), 'external_test_evaluated': False})
        raw_images = extractor.prepare_images(frame.iloc[indices].path.tolist())
        cache_images = extractor.prepare_images(frame.iloc[indices].cache_path.tolist(), True)
        assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(raw_images, cache_images))
        live_base, live_large, live_bank = extractor.extract_patch_preprocessed(raw_images)
        cache_base = {name: base[name][indices] for name in KEYS}
        cache_large = {name: large[name][indices] for name in KEYS}
        cache_large['side_tokens'] = tokens['registers'][indices]
        cache_bank = {name: np.asarray(value[indices]) for name, value in expected_bank.items()}
        feature_errors = {}
        for branch, live, cached in [('B', live_base, cache_base), ('L', live_large, cache_large)]:
            for name in live:
                assert np.allclose(live[name], cached[name], atol=.05, rtol=.005), (branch, name)
                feature_errors[branch+'_'+name] = float(np.abs(live[name]-cached[name]).max())
        for name in BANK_KEYS:
            assert np.array_equal(live_bank[name], cache_bank[name]), (name, indices.tolist())
            assert live_bank[name].dtype == cache_bank[name].dtype
        adapters = []
        for arm in arms:
            model, source = restore_reviewed_adapter('register_patch_joint', arm, 'validation', 1)
            with torch.inference_mode():
                from register_patch_joint_experiments import predict
                reference = predict(model, source, {**large, 'side_tokens': tokens['registers']},
                    expected_bank, indices)
                from_cache = adapter_probability(model, source, cache_large, cache_bank)
                assert np.array_equal(from_cache, reference), 'The new API must match original engine on identical cached tensors.'
                from_raw = adapter_probability(model, source, live_large, live_bank)
                assert np.allclose(from_raw, from_cache, atol=5e-4, rtol=.002)
                assert np.array_equal(from_raw.argmax(1), from_cache.argmax(1))
            matches = [row for row in checkpoint_rows
                if row['arm'] == arm and row['phase'] == 'validation' and row['seed'] == 1]
            assert len(matches) == 1
            adapters.append({'arm': arm, 'phase': 'validation', 'seed': 1,
                'checkpoint_sha256': matches[0]['checkpoint_sha256'],
                'new_api_cached_probability_bitwise_matches_original_engine': True,
                'raw_image_vs_cached_probability_maximum_error': float(np.abs(from_raw-from_cache).max()),
                'raw_image_vs_cached_decisions_identical': True})
            del model
            torch.cuda.empty_cache()
        cases.append({'manifest_indices': indices.tolist(),
            'raw_preprocessed_pixels_equal_stored_cache': True,
            'all_six_patch_arrays_bitwise_match_original_extraction_batch': True,
            'feature_maximum_errors': feature_errors, 'adapter_checks': adapters})
        print(json.dumps({'event': 'patch_image_batch_verified', 'manifest_indices': indices.tolist(),
            'six_patch_arrays_bitwise_identical': True, 'adapter_checks': len(adapters)}), flush=True)
    for section in ['code_sha256', 'frozen_artifacts', 'image_sha256']:
        for name, expected in protocol[section].items():
            assert sha(name) == expected, name
    dump(DEST, {'passed': True, 'protocol_sha256': sha(OUT/'protocol.json'),
        'manifest_sha256': digest, 'cases': cases, 'original_images_checked': 24,
        'source_adapter_models_checked': 4, 'adapter_batch_checks': 12,
        'checks': ['Original image fixed crop/pad/JPEG pixels match stored cache exactly',
            'Each top64 bank array matches original extraction bitwise with matching batch boundaries',
            'All eight B/L classification streams plus four-register stream match inherited feature tolerances',
            'New inference API exactly matches reviewed original adapter engine on identical cached tensors',
            'Original-image and cached-adapter probabilities agree within fixed engineering tolerance with identical decisions',
            'Only completed independently verified checkpoints restore, with ordinary CUDA parameters'],
        'feature_tolerance': {'atol': .05, 'rtol': .005},
        'raw_vs_cached_probability_tolerance': {'atol': 5e-4, 'rtol': .002},
        'scope_limit': 'Engineering numerical equivalence check on existing development images only; no accuracy or new candidate ranking. Does not confirm an adapter gain or create a promoted model. Different image batch shapes may change float16 top64 selection near ties; exact bank equality is asserted here only at original eight-image boundaries.',
        'new_model_fits': 0, 'new_candidate_scores': 0, 'external_images_read': 0,
        'external_test_evaluated': False, 'seconds': time.time()-started,
        'code_sha256': protocol['code_sha256']})
    dump(OUT/'state.json', {'status': 'completed', 'pid': os.getpid(), 'passed': True,
        'external_test_evaluated': False})
    print('Original-image patch feature and reviewed-adapter equivalence checks passed.', flush=True)


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
