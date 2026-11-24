"""Independent restoration of nine packaged models and fixed register fusion."""
import json
from datetime import datetime, timezone
import numpy as np
import torch
from experiment import ROOT, load_data, metrics, dump
from kernel_residual_experiments import sha
from large_token_data import load_all
from normal_subspace_experiments import predict_cpu as base_predict
from audit_large_token_evidence import independent_probability
from package_register_blend import checkpoint_path


def main():
    torch.set_num_threads(4); folder = ROOT/'results/milestone8_register_blend'
    package = json.loads((folder/'package.json').read_text()); selection = json.loads((folder/'selection_development_only.json').read_text())
    assert selection['development_and_fresh_patient_confirmed'] and selection['original_image_inference_verified']
    for file, key in [('package.json', 'package_sha256'), ('inference_verification.json', 'inference_verification_sha256'), ('cli_verification.json', 'cli_verification_sha256')]:
        assert sha(folder/file) == selection[key]
    for name in ['inference_verification.json', 'cli_verification.json']: assert json.loads((folder/name).read_text())['passed']
    for name, expected in package['code_sha256'].items(): assert sha(ROOT/name) == expected
    for key in ['selection_record', 'confirmation_record', 'confirmation_verification', 'confirmation_integrity']:
        assert sha(ROOT/package[key]) == package[key+'_sha256']
    confirmation = json.loads((ROOT/package['confirmation_record']).read_text())
    assert confirmation['confirmation_gate_passed'] and not confirmation['external_test_evaluated']
    candidate = next(row for row in confirmation['arms'] if row['arm'] == 'candidate')
    assert set(confirmation['paired_candidate_vs_controls']) == {'m7', 'm6', 'reference', 'fixed_gaussian'}
    for arm, comparison in confirmation['paired_candidate_vs_controls'].items():
        assert comparison['macro_f1_ci95'][0] > 0
        assert candidate['mean_macro_f1'] > next(row['mean_macro_f1'] for row in confirmation['arms'] if row['arm'] == arm)
    frame, large, tokens, digest = load_all(); _, base, base_digest = load_data('multiscale_vit')
    assert base_digest == digest == package['manifest_sha256']; large['side_tokens'] = tokens['registers']; del tokens
    ids = np.flatnonzero(frame.split.eq('val')); checked, components = [], {}
    for branch in ['reference', 'subspace', 'token']:
        features = large if branch == 'token' else base; ps = []
        for seed in range(3):
            source = checkpoint_path(branch, seed); path = folder/'frozen_package'/f'seed_{seed}'/(branch+'.pt')
            assert sha(path) == sha(source) == package['model_sha256'][f'{branch}_seed{seed}']
            assert package['source_model_sha256'][str(source)] == sha(source)
            c = torch.load(path, map_location='cpu', weights_only=False)
            assert c['manifest_sha256'] == digest and c['cfg']['seed'] == seed
            if branch == 'token':
                assert c['cfg']['token_evidence_stream'] == 'registers'
                p = independent_probability(c, features, ids)
            else: p = base_predict(c, features, ids, independent=True)
            saved = np.load(source.parent/'val_predictions.npz')['p']
            assert np.allclose(p, saved, atol=6e-5, rtol=6e-5) and np.array_equal(p.argmax(1), saved.argmax(1))
            ps.append(p); checked.append({'branch': branch, 'seed': seed, 'checkpoint_sha256': sha(path),
                'maximum_independent_probability_error': float(np.abs(p-saved).max())})
        components[branch] = np.mean(ps, 0).astype('float64')
    expected = {'reference': .1875, 'subspace': .5625, 'token': .25}; assert package['branch_weights'] == expected
    p = sum(expected[key]*value for key, value in components.items()).astype('float32')
    chosen = json.loads((ROOT/package['selection_record']).read_text())['candidate']
    saved = np.load(ROOT/'runs/large_token_evidence'/chosen['name']/'validation/predictions.npz')
    assert np.array_equal(saved['indices'], ids) and np.array_equal(p.argmax(1), saved['pred'])
    assert np.allclose(p, saved['p'], atol=6e-5, rtol=6e-5)
    measured = metrics(frame.iloc[ids].label, p)
    for key in ['accuracy', 'macro_f1', 'balanced_accuracy', 'confusion_matrix', 'recall']:
        assert measured[key] == package['development_validation'][key]
    assert len(checked) == 9
    dump(folder/'independent_package_audit.json', {'passed': True, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'package_sha256': sha(folder/'package.json'), 'selection_sha256': sha(folder/'selection_development_only.json'),
        'manifest_sha256': digest, 'actual_packaged_heads_checked': 9, 'validation_images_checked': len(ids),
        'validation': measured, 'branch_weights': expected, 'fits': checked,
        'maximum_three_branch_validation_error': float(np.abs(p-saved['p']).max()),
        'fresh_confirmation_all_four_comparisons_passed': True, 'frozen_original_image_and_cli_verifications_checked': True,
        'source_and_package_bytes_identical': True, 'code_sha256': sha(ROOT/'audit_register_package.py'), 'external_test_evaluated': False})
    print(json.dumps({'passed': True, 'heads': 9, 'validation': measured, 'external_test_evaluated': False}), flush=True)


if __name__ == '__main__': main()
