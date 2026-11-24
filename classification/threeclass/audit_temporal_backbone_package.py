"""Independently recover the fifteen packaged heads and their fixed five-branch rule."""
import json
from datetime import datetime, timezone
import numpy as np
import torch
from experiment import ROOT, KEYS, Model, load_data, normalize, metrics, dump
from kernel_residual_experiments import sha
from normal_subspace import independent_transform
from normal_cross_geometry import independent_features
from temporal_backbone_mixtures import checkpoint_path


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    folder = ROOT/'results/milestone8_temporal_backbone'
    package = json.loads((folder/'package.json').read_text())
    selection = json.loads((folder/'selection_development_only.json').read_text())
    assert selection['development_and_fresh_patient_confirmed'] and selection['original_image_inference_verified']
    assert sha(folder/'package.json') == selection['package_sha256']
    assert sha(folder/'inference_verification.json') == selection['inference_verification_sha256']
    assert sha(folder/'cli_verification.json') == selection['cli_verification_sha256']
    for name in ['inference_verification.json', 'cli_verification.json']:
        assert json.loads((folder/name).read_text())['passed']
    for name, expected in package['code_sha256'].items(): assert sha(ROOT/name) == expected
    for key in ['selection_record', 'confirmation_record', 'confirmation_verification', 'confirmation_integrity']:
        assert sha(ROOT/package[key]) == package[key+'_sha256']
    confirmation = json.loads((ROOT/package['confirmation_record']).read_text())
    assert confirmation['confirmation_gate_passed'] and not confirmation['external_test_evaluated']
    candidate = next(row for row in confirmation['arms'] if row['arm'] == 'candidate')
    for control, comparison in confirmation['paired_candidate_vs_controls'].items():
        assert comparison['macro_f1_ci95'][0] > 0
        assert candidate['mean_macro_f1'] > next(row['mean_macro_f1'] for row in confirmation['arms'] if row['arm'] == control)
    frame, base, digest = load_data('multiscale_vit')
    assert digest == package['manifest_sha256']
    large_protocol = json.loads((ROOT/'runs/dinov3_large/protocol.json').read_text())
    with np.load(large_protocol['feature_storage']) as bank:
        assert np.array_equal(bank['indices'], np.arange(2091)) and str(bank['manifest_sha256']) == digest
        large = {key: bank[key].astype('float32') for key in KEYS}
    ids = np.flatnonzero(frame.split.eq('val')); components = {}; checked = []
    for branch in ['reference', 'subspace', 'ema', 'large', 'dropout']:
        data = {key: np.concatenate([base[key][:2091], large[key]], -1) for key in KEYS} if branch == 'dropout' else (large if branch == 'large' else base)
        ps = []
        for seed in range(3):
            path = folder/'frozen_package'/f'seed_{seed}'/(branch+'.pt')
            source = checkpoint_path('validation', branch, seed)
            assert sha(path) == sha(source) == package['model_sha256'][f'{branch}_seed{seed}']
            assert package['source_model_sha256'][str(source)] == sha(source)
            c = torch.load(path, map_location='cpu', weights_only=False)
            assert c['manifest_sha256'] == digest and c['cfg']['seed'] == seed
            state = c['state_dict']
            excluded = ('base_center_', 'base_basis_', 'large_center_', 'large_basis_') if branch == 'dropout' else ('normal_',)
            plain = Model(c['cfg'], c['dims'], len(c['diseases'])).eval()
            plain.load_state_dict({key: value for key, value in state.items() if not key.startswith(excluded)}, strict=True)
            batches = []
            for start in range(0, len(ids), 64):
                normalized, _ = normalize({key: value[ids[start:start+64]] for key, value in data.items()}, None, c['stats'])
                if branch == 'dropout':
                    normalized = independent_features(normalized, state)
                elif branch != 'reference':
                    normalized = {key: independent_transform(value, state['normal_center_'+key].numpy(),
                        state['normal_basis_'+key].numpy(), 1.) for key, value in normalized.items()}
                batches.append(plain({key: torch.from_numpy(value) for key, value in normalized.items()})['p'].numpy())
            p = np.concatenate(batches); saved = np.load(source.parent/'val_predictions.npz')['p']
            assert np.allclose(p, saved, atol=6e-5, rtol=6e-5) and np.array_equal(p.argmax(1), saved.argmax(1))
            ps.append(p); checked.append({'branch': branch, 'seed': seed, 'checkpoint_sha256': sha(path),
                'maximum_independent_probability_error': float(np.abs(p-saved).max())})
        components[branch] = np.mean(ps, 0).astype('float64')
    expected = {'reference': .21875, 'subspace': .140625, 'ema': .140625, 'large': .125, 'dropout': .375}
    assert package['dropout_mixture_weight'] == .5 and package['branch_weights'] == expected
    p = sum(expected[key]*value for key, value in components.items()).astype('float32')
    selected = json.loads((ROOT/package['selection_record']).read_text())['candidate']
    saved = np.load(ROOT/'runs/temporal_backbone_mixtures'/selected['name']/'validation/predictions.npz')
    assert np.array_equal(saved['indices'], ids) and np.allclose(p, saved['p'], atol=6e-5, rtol=6e-5)
    assert np.array_equal(p.argmax(1), saved['pred'])
    measured = metrics(frame.iloc[ids].label, p)
    for key in ['accuracy', 'macro_f1', 'balanced_accuracy', 'confusion_matrix', 'recall']:
        assert measured[key] == package['development_validation'][key]
    assert len(checked) == 15 and abs(sum(expected.values())-1.) < 1e-12
    dump(folder/'independent_package_audit.json', {'passed': True, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'package_sha256': sha(folder/'package.json'), 'selection_sha256': sha(folder/'selection_development_only.json'),
        'manifest_sha256': digest, 'actual_packaged_heads_checked': 15, 'validation_images_checked': len(ids),
        'validation': measured, 'branch_weights': expected, 'fits': checked,
        'maximum_five_branch_validation_error': float(np.abs(p-saved['p']).max()),
        'fresh_confirmation_all_four_comparisons_passed': True, 'frozen_original_image_and_cli_verifications_checked': True,
        'source_and_package_bytes_identical': True, 'code_sha256': sha(ROOT/'audit_temporal_backbone_package.py'),
        'external_test_evaluated': False})
    print(json.dumps({'passed': True, 'heads': 15, 'validation': measured, 'external_test_evaluated': False}), flush=True)


if __name__ == '__main__': main()
