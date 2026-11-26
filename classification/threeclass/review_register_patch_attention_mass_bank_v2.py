"""Root saved-array review of versioned full attention normalization; no model forward."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'runs/register_patch_attention_mass_bank_v2'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(part)
    return digest.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def main():
    destination = ROOT/'results/register_patch_attention_mass_bank_v2_root_review.json'
    assert not destination.exists()
    proof = json.loads((OUT/'verification.json').read_text())
    protocol = json.loads((OUT/'protocol.json').read_text())
    assert json.loads((OUT/'state.json').read_text())['status'] == 'completed'
    assert proof['passed'] and proof['development_images'] == 2091
    assert sha(OUT/'verification.json') == '5e41cd49edf5d8e8c8de7dba6a93dc02722afef3d3e9d88a96d74704cdb4e807'
    assert sha(OUT/'protocol.json') == proof['protocol_sha256']
    checked = {str(OUT/'verification.json'): sha(OUT/'verification.json'), str(OUT/'protocol.json'): sha(OUT/'protocol.json')}
    for section in ['code_sha256', 'frozen_artifacts', 'cache_image_sha256']:
        for name, expected in protocol[section].items():
            assert sha(name) == expected, name
            checked[name] = expected
    for name, expected in proof['frozen_artifacts'].items():
        assert sha(name) == expected, name
        checked[name] = expected
    arrays = {name: np.load(OUT/(name+'.npy'), mmap_mode='r') for name in ['selected_mass', 'prefix_mass']}
    assert all(a.shape == (2091, 2, 16, 4) and a.dtype == np.float64 and np.isfinite(a).all() for a in arrays.values())
    assert (arrays['selected_mass'] > 0).all() and (arrays['prefix_mass'] >= 0).all()
    assert (arrays['selected_mass']+arrays['prefix_mass'] <= 1+1e-12).all()
    coverage = []
    for begin in range(0, 2091, 8):
        row = json.loads((OUT/'chunks'/f'{begin:05d}.json').read_text())
        ids = np.asarray(row['indices'])
        assert np.array_equal(ids, np.arange(begin, min(begin+8, 2091)))
        assert row['all_six_source_bank_arrays_bitwise_equal']
        for name in arrays:
            assert array_sha(arrays[name][ids]) == row['array_sha256'][name]
        coverage.extend(ids.tolist())
    assert coverage == list(range(2091))
    bank = ROOT/'runs/register_patch_query_bank/arrays'
    cases = []
    for row in proof['independent_cpu_witnesses']:
        with np.load(row['path']) as data:
            q = data['query'].astype(np.float64)
            k = data['all_keys'].astype(np.float64)
            ids = data['indices']; view = row['view']
            assert ids.tolist() == row['indices'] and sha(row['path']) == row['sha256']
            positions = data['positions'].astype(np.int64)
            assert np.array_equal(data['query'], np.load(bank/'query_raw.npy', mmap_mode='r')[ids, view])
            assert np.array_equal(positions, np.load(bank/'patch_indices.npy', mmap_mode='r')[ids, view])
            chosen_keys = np.take_along_axis(k[:, :, 5:], positions[..., None], axis=2)
            assert np.array_equal(chosen_keys, np.load(bank/'patch_keys.npy', mmap_mode='r')[ids, view])
            logits = np.einsum('bhqd,bhkd->bhqk', q, k, optimize=False)/8
            numerator = np.exp(logits-logits.max(-1, keepdims=True))
            full = numerator/numerator.sum(-1, keepdims=True)
            selected = np.take_along_axis(full, positions[:, :, None, :]+5, axis=-1)
            mass = selected.sum(-1); prefix = full[..., :5].sum(-1)
            err = max(float(np.abs(mass-data['selected_mass']).max()), float(np.abs(prefix-data['prefix_mass']).max()))
            assert np.allclose(mass, data['selected_mass'], atol=5e-12, rtol=5e-12)
            assert np.allclose(prefix, data['prefix_mass'], atol=5e-12, rtol=5e-12)
            for name in arrays:
                assert np.array_equal(data[name], arrays[name][ids, view])
            local_logits = np.einsum('bhqd,bhkd->bhqk', q, chosen_keys, optimize=False)/8
            local_exp = np.exp(local_logits-local_logits.max(-1, keepdims=True))
            conditional = local_exp/local_exp.sum(-1, keepdims=True)
            assert np.allclose(conditional*mass[..., None], selected, atol=1e-12, rtol=1e-11)
            cases.append({'indices': ids.tolist(), 'view': view, 'independent_cpu_maximum_error': err,
                'selected_mass_minimum': float(mass.min()), 'selected_mass_maximum': float(mass.max()),
                'saved_q_and_selected_k_and_indices_match_original_bank': True})
    assert len(cases) == 6 and {r['view'] for r in cases} == {0, 1}
    assert proof['backbone_view_forwards'] == 530
    assert proof['classification_fits'] == proof['classification_scores'] == proof['classifier_probability_predictions'] == proof['external_images_read'] == 0
    record = {'passed': True, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'manifest_sha256': proof['manifest_sha256'], 'input_artifact_sha256': checked,
        'reviewer_sha256': sha(__file__), 'chunks_checked': 262, 'development_images': 2091,
        'saved_cpu_witnesses': cases, 'original_builder_backbone_view_forwards': 530,
        'review_new_model_forwards': 0, 'classification_fits': 0, 'classification_scores': 0,
        'external_images_read': 0, 'review_scope': 'Hash/inventory/chunk coverage and saved six full-Q/K CPU algebra reconstruction. The builder performed530 backbone forwards and bitwise six-array reproduction/fresh mass replay; this review adds no new source-model replay. Full float64 arithmetic from captured FP16 Q/K; no source fused-attention bitwise claim.'}
    with destination.open('x') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'passed': True, 'checked_files': len(checked), 'cpu_witnesses': len(cases), 'maximum_error': max(r['independent_cpu_maximum_error'] for r in cases), 'sha256': sha(destination)}))


if __name__ == '__main__':
    main()
