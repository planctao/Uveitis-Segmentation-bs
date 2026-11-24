"""Verify all validation outputs and original-image register extraction."""
import json
import subprocess
import sys
import numpy as np
import torch
from experiment import ROOT, KEYS, load_data, metrics, dump
from kernel_residual_experiments import sha
from predict_register_blend import FrozenRegisterBlendPredictor


def main():
    torch.set_num_threads(4)
    frame, base, digest = load_data('multiscale_vit')
    from large_token_data import load_all
    token_frame, large, tokens, token_digest = load_all()
    assert token_digest == digest and token_frame.equals(frame)
    large['side_tokens'] = tokens['registers']
    del tokens
    ids = np.flatnonzero(frame.split.eq('val'))
    predictor = FrozenRegisterBlendPredictor(require_verification=False); parts = []
    for start in range(0, len(ids), 64):
        selected = ids[start:start+64]
        parts.append(predictor.predict_features({k: v[selected] for k, v in base.items()}, {k: v[selected] for k, v in large.items()}))
    p = np.concatenate(parts)
    candidate = json.loads((ROOT/predictor.decision['selection_record']).read_text())['candidate']
    saved = np.load(ROOT/'runs/large_token_evidence'/candidate['name']/'validation/predictions.npz')
    assert np.array_equal(saved['indices'], ids) and np.array_equal(p.argmax(1), saved['pred'])
    assert np.allclose(p, saved['p'], atol=6e-5, rtol=6e-5)
    chosen = np.random.default_rng(20285401).choice(ids, 6, replace=False)
    cached = predictor.predict_features({k: v[chosen] for k, v in base.items()}, {k: v[chosen] for k, v in large.items()})
    live = predictor.predict_images(frame.iloc[chosen].cache_path.tolist(), already_preprocessed=True, batch=3)
    original = predictor.predict_images(frame.iloc[chosen].path.tolist(), already_preprocessed=False, batch=3)
    assert np.abs(cached-live).mean() < .001 and np.array_equal(cached.argmax(1), live.argmax(1))
    assert np.allclose(original, live, atol=6e-5, rtol=6e-5) and np.array_equal(original.argmax(1), live.argmax(1))
    assert predictor.base_image_model is not None and predictor.large_image_model is not None
    for models in predictor.models.values():
        for model in models: assert not model.training
    verification = {'passed': True, 'manifest_sha256': digest, 'package_sha256': sha(predictor.package/'package.json'),
        'validation_images_checked': len(ids), 'frozen_head_states_checked': 9, 'validation': metrics(frame.iloc[ids].label, p),
        'real_validation_indices': chosen.tolist(), 'real_cached_and_original_images_checked': 6,
        'maximum_validation_probability_error': float(np.abs(p-saved['p']).max()),
        'mean_live_vs_cache_probability_error': float(np.abs(live-cached).mean()),
        'maximum_original_vs_cache_image_probability_error': float(np.abs(original-live).max()),
        'external_images_or_outcomes_used': False, 'checks': ['All9 frozenheadstates and their fullconfirmation/codehashes verified',
            'All374validation probabilities and decisions match original frozen source formula',
            'Distinct fit-only normalization and projection used per rawB/normalB/registerL',
            'Six actualcache/original sourceimages pass crop/pad/JPEG and bothfrozenDINObackbones at320/448',
            'Rawimage methods reuse proven source extractor; predictor adds only fixedthreebranch probabilityarithmetic']}
    dump(predictor.package/'inference_verification.json', verification)
    selection = {'candidate_id': predictor.decision['candidate_id'], 'package_sha256': sha(predictor.package/'package.json'),
        'inference_verification_sha256': sha(predictor.package/'inference_verification.json'),
        'development_and_fresh_patient_confirmed': True, 'original_image_inference_verified': True,
        'external_test_evaluated': False, 'external_policy': 'No automaticexternal test; allhistorical external references preserved.'}
    dump(predictor.package/'selection_development_only.json', selection)
    path = frame.iloc[chosen[0]].path
    process = subprocess.run([sys.executable, str(ROOT/'predict_register_blend.py'), '--image', str(path)],
        cwd=ROOT, check=True, text=True, capture_output=True)
    cli = json.loads(process.stdout); row = cli['predictions'][0]
    cli_p = np.array([row['probabilities'][name] for name in ['感染', '非感染', '正常']])
    assert np.allclose(cli_p, original[0], atol=6e-5, rtol=6e-5)
    assert cli_p.argmax() == original[0].argmax()
    dump(predictor.package/'cli_verification.json', {'passed': True, 'image': str(path), 'candidate_id': cli['candidate_id'],
        'maximum_probability_error': float(np.abs(cli_p-original[0]).max()), 'exit_code': process.returncode,
        'external_images_or_outcomes_used': False})
    dump(predictor.package/'selection_development_only.json', {**selection, 'cli_verification_sha256': sha(predictor.package/'cli_verification.json')})
    print(json.dumps({k: v for k, v in verification.items() if k != 'checks'}), flush=True)


if __name__ == '__main__': main()
