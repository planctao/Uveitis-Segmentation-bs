"""Actual-image depth token extraction with independent prefix indexing."""
import json
import numpy as np
import pandas as pd
import torch
from experiment import ROOT, dump
from kernel_residual_experiments import sha
from extract_dinov3_large import Images, make_model
from register_depth_features import TOKEN_KEYS, forward_depth_features, install_depth_hooks


@torch.inference_mode()
def main():
    torch.set_num_threads(4); torch.manual_seed(20261124)
    protocol = json.loads((ROOT/'runs/dinov3_large/protocol.json').read_text()); frame = pd.read_csv(ROOT/'metadata/manifest.csv')
    digest = sha(ROOT/'metadata/manifest.csv'); assert digest == protocol['manifest_sha256']
    with np.load(protocol['feature_storage']) as bank: prior = {key: bank[key] for key in ['global', 'regions', 'mid', 'dispersion']}
    token_proof = json.loads((ROOT/'verification_large_token_feature_bank.json').read_text())
    assert token_proof['passed'] and sha(token_proof['feature_storage']) == token_proof['feature_bank_sha256']
    with np.load(token_proof['feature_storage']) as bank: registers = bank['registers']
    model = make_model(protocol); captured = {}; hooks = install_depth_hooks(model, captured); final = {}
    hooks.append(model.norm.register_forward_hook(lambda module, args, output: final.update(tokens=output)))
    chosen = [0, 352, 887, 1501, 1800, 2090]; dataset = Images(frame, chosen); checked = []
    for pos, index in enumerate(chosen):
        actual, a, b = dataset[pos]; assert actual == index; views, old, errors = [], [], []
        for image in [a, b]:
            values, legacy = forward_depth_features(model, image[None].cuda(), captured)
            tensors = [captured['early'], captured['middle'], final['tokens']]
            independent = {key: torch.index_select(value, 1, torch.tensor([1, 2, 3, 4], device=value.device)).float().cpu().numpy().astype('float16' if key == 'final_registers' else 'float32') for key, value in zip(TOKEN_KEYS, tensors)}
            for key in TOKEN_KEYS:
                assert values[key].shape == (1, 4, 1024) and np.isfinite(values[key]).all()
                assert np.array_equal(values[key], independent[key]), key
            errors.append({key: float(np.abs(values[key].astype('float32')-independent[key].astype('float32')).max()) for key in TOKEN_KEYS})
            views.append(values); old.append(legacy)
        legacy_errors = {}
        for key in prior:
            actual = np.concatenate([view[key] for view in old], -1)[0].astype('float32'); expected = prior[key][index].astype('float32')
            assert np.allclose(actual, expected, atol=.05, rtol=.005), key
            legacy_errors[key] = float(np.abs(actual-expected).max())
        final_values = np.concatenate([view['final_registers'] for view in views], -1)[0].astype('float32')
        assert np.allclose(final_values, registers[index], atol=.05, rtol=.005)
        checked.append({'development_index': index, 'cache_sha256': sha(frame.iloc[index].cache_path),
            'independent_prefix_errors_per_scale': errors, 'legacy_feature_errors': legacy_errors,
            'maximum_absolute_depth_activations': {key: max(float(np.abs(view[key]).max()) for view in views) for key in TOKEN_KEYS},
            'final_register_source_error': float(np.abs(final_values-registers[index]).max())})
    for handle in hooks: handle.remove()
    dump(ROOT/'verification_register_depth_preflight.json', {'passed': True, 'manifest_sha256': digest,
        'actual_images_recomputed': checked, 'depths': ['raw_block7', 'raw_block15', 'final_norm'],
        'token_counts': {key: 4 for key in TOKEN_KEYS}, 'scales': [320, 448],
        'token_dtypes': {key: 'float16' if key == 'final_registers' else 'float32' for key in TOKEN_KEYS},
        'pretrained_weights_sha256': protocol['pretrained_weights']['sha256'],
        'code_sha256': {name: sha(ROOT/name) for name in ['register_depth_features.py', 'verify_register_depth_preflight.py', 'extract_dinov3_large.py']},
        'checks': ['Allthree4registerstreams recovered exactly with independentpositionindexing on6actualdevelopmentimages/twoscales',
            'Allhistorical CLS/regions/block15mean/patchstd plusfinalregister source agree within existingAMPtolerance',
            'Samefrozenbackbone/transforms; noclassificationfits/scores or externalimages/outcomes'],
        'classification_fits': 0, 'classification_scores': 0, 'external_test_evaluated': False})
    print('Six real images: three register depths exactly match independent indexing; all legacy sources agree.', flush=True)


if __name__ == '__main__': main()
