"""Explicit image-level register slicing and cache feature verification."""
import json
import numpy as np
from PIL import Image
import torch
from experiment import ROOT, KEYS, load_data, dump
from large_token_data import load_all
from predict_register_blend import FrozenRegisterBlendPredictor
from kernel_residual_experiments import sha


def main():
    torch.set_num_threads(4)
    frame, large, tokens, digest = load_all(); _, base, other_digest = load_data('multiscale_vit'); assert digest == other_digest
    ids = np.random.default_rng(20285402).choice(np.flatnonzero(frame.split.eq('val')), 6, replace=False)
    predictor = FrozenRegisterBlendPredictor(); cases = []
    for index in ids:
        with Image.open(frame.iloc[index].cache_path) as source: image = source.convert('RGB')
        live_base, live_large = predictor.extract_preprocessed([image])
        errors = {}
        for key in KEYS:
            expected = large[key][index:index+1]
            assert np.allclose(live_large[key], expected, atol=.05, rtol=.005), key
            errors['large_'+key] = float(np.abs(live_large[key]-expected).max())
            assert np.allclose(live_base[key], base[key][index:index+1], atol=.05, rtol=.005), key
            errors['base_'+key] = float(np.abs(live_base[key]-base[key][index:index+1]).max())
        registers = tokens['registers'][index:index+1]
        assert live_large['side_tokens'].shape == registers.shape == (1, 4, 2048)
        assert np.allclose(live_large['side_tokens'], registers, atol=.05, rtol=.005)
        direct = []
        for transform in predictor.transforms:
            image_tensor = transform(image)[None].to(predictor.device)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                sequence = predictor.large_image_model.forward_features(image_tensor)
            # Independent position indexing excludes CLS and every spatial token.
            direct.append(torch.index_select(sequence, 1, torch.tensor([1, 2, 3, 4], device=sequence.device)).to(torch.float16).float().cpu().numpy())
        direct = np.concatenate(direct, -1)
        assert np.array_equal(direct, live_large['side_tokens'])
        errors['register_cache'] = float(np.abs(registers-live_large['side_tokens']).max())
        cases.append({'validation_index': int(index), 'maximum_feature_errors': errors, 'direct_four_prefix_positions_identical': True})
    dump(predictor.package/'image_feature_verification.json', {'passed': True, 'manifest_sha256': digest,
        'package_sha256': sha(predictor.package/'package.json'), 'six_real_validation_images': cases,
        'tolerance': 'Same frozen backbone numerical tolerance as prior feature-bank audit:atol.05/rtol.005; directregisterposition indexing must matchexactly.',
        'all_eight_original_streams_and_new_register_stream_checked': True, 'external_images_or_outcomes_used': False,
        'code_sha256': sha(ROOT/'verify_register_image_features.py')})
    print('Six images: all B/L features match frozen caches; direct register positions match exactly.', flush=True)


if __name__ == '__main__': main()
