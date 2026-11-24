"""Test predeclared constant probability mixtures using development predictions."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiment import ROOT, metrics, dump
from milestone2 import paired_oof


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def combine(reference, donor, weight):
    assert 0 <= weight <= 1 and reference.shape == donor.shape
    for p in [reference, donor]:
        assert p.ndim == 2 and p.shape[1] == 3 and np.isfinite(p).all()
        assert (p >= 0).all() and np.allclose(p.sum(1), 1, atol=1e-5)
    return ((1 - weight) * reference + weight * donor).astype('float32')


def validation(record, frame):
    sources = [Path(checkpoint).parent / 'val_predictions.npz' for checkpoint in record['checkpoints']]
    p = np.mean([np.load(path)['p'] for path in sources], 0)
    m = metrics(frame.loc[frame.split.eq('val'), 'label'], p)
    assert abs(m['macro_f1'] - record['val']['macro_f1']) < 1e-12
    return p, {str(path): sha(path) for path in sources}


def main():
    out = ROOT / 'runs/resolution_probability_fusion'
    protocol = json.loads((out / 'protocol.json').read_text())
    verification = json.loads((ROOT / 'results/research_raw_highres/verification.json').read_text())
    assert verification['passed'] and verification['full_head_inference_reproduced']
    frame = pd.read_csv(ROOT / 'metadata/manifest.csv')
    digest = sha(ROOT / 'metadata/manifest.csv')
    frozen = json.loads((ROOT / protocol['reference']).read_text())
    reference = frozen['selected']
    reference_cv = Path(reference['source_cv_summary']).parent
    donors = json.loads((ROOT / 'runs/raw_highres/leaderboard.json').read_text())
    assert [record['feature'] for record in donors] == protocol['donor_features']
    rp, source_hashes = validation(reference, frame)
    original = np.flatnonzero(frame.split.eq('train').to_numpy())
    results = []
    for donor in donors:
        dp, hashes = validation(donor, frame)
        source_hashes.update(hashes)
        donor_cv = ROOT / 'runs' / donor['feature'] / 'candidate_crossvalidation' / ('continuous_' + donor['candidate_id'])
        for weight in protocol['donor_weights']:
            identity = {'reference': reference['candidate_id'], 'donor': donor['candidate_id'], 'weight': weight}
            candidate_id = 'experts_' + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
            directory = out / candidate_id
            p = combine(rp, dp, weight)
            record = {'candidate_id': candidate_id, 'architecture': 'fixed_probability_experts',
                'donor_feature': donor['feature'], 'donor_weight': weight,
                'components': [{'feature': reference['feature'], 'weight': 1 - weight,
                    'checkpoints': reference['checkpoints'], 'candidate_id': reference['candidate_id']},
                    {'feature': donor['feature'], 'weight': weight,
                    'checkpoints': donor['checkpoints'], 'candidate_id': donor['candidate_id']}],
                'val': metrics(frame.loc[frame.split.eq('val'), 'label'], p),
                'new_seed_fits': 0, 'external_test_evaluated': False}
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(directory / 'val_ensemble.npz', p=p, pred=p.argmax(1))
            folds = []
            seen = []
            for fold in range(3):
                a = reference_cv / f'fold_{fold}/outer_predictions.npz'
                b = donor_cv / f'fold_{fold}/outer_predictions.npz'
                source_hashes.update({str(path): sha(path) for path in [a, b]})
                left, right = np.load(a), np.load(b)
                indices = left['indices']
                assert np.array_equal(indices, right['indices'])
                assert frame.iloc[indices].split.eq('train').all()
                p = combine(left['p'], right['p'], weight)
                target = directory / f'fold_{fold}'
                target.mkdir(exist_ok=True)
                np.savez_compressed(target / 'outer_predictions.npz', indices=indices, p=p, pred=p.argmax(1))
                result = {'fold': fold, 'metrics': metrics(frame.iloc[indices].label, p)}
                dump(target / 'metrics.json', result)
                folds.append(result)
                seen.extend(indices.tolist())
            assert sorted(seen) == original.tolist()
            record['cv'] = {'mean_macro_f1': float(np.mean([f['metrics']['macro_f1'] for f in folds])),
                'sd_macro_f1': float(np.std([f['metrics']['macro_f1'] for f in folds], ddof=1)),
                'mean_accuracy': float(np.mean([f['metrics']['accuracy'] for f in folds]))}
            record['paired_original_oof_versus_milestone5'] = paired_oof(reference_cv, directory, frame)
            record['development_gate_passed'] = record['cv']['mean_macro_f1'] > frozen['cv']['mean_macro_f1'] and record['val']['macro_f1'] >= reference['val']['macro_f1']
            dump(directory / 'candidate.json', record)
            results.append(record)
            print(json.dumps({k: record[k] for k in ['candidate_id', 'donor_feature', 'donor_weight', 'val', 'cv', 'development_gate_passed']}), flush=True)
    assert len(results) == 12
    for record in results:
        if '_raw' not in record['donor_feature']:
            continue
        matched = next(row for row in results if row['donor_feature'] == record['donor_feature'].replace('_raw', '_up') and row['donor_weight'] == record['donor_weight'])
        record['paired_original_oof_versus_matched_upsampling'] = paired_oof(out / matched['candidate_id'], out / record['candidate_id'], frame)
        dump(out / record['candidate_id'] / 'candidate.json', record)
    eligible = [row for row in results if row['development_gate_passed']]
    report = {'manifest_sha256': digest, 'protocol': protocol, 'candidates': results,
        'source_prediction_sha256': source_hashes, 'completed_composite_candidates': len(results),
        'new_seed_fits': 0, 'external_files_read': False, 'external_test_evaluated': False,
        'decision': 'Run predeclared repeated-patient confirmation for the strongest eligible candidate.' if eligible else 'Keep frozen milestone5; no combination met both development gates.',
        'inference_limit': 'Exploratory patient bootstrap on the original development folds; intervals do not correct prior search or all training-fit uncertainty.'}
    destination = ROOT / 'results/research_probability_fusion'
    dump(destination / 'comparison.json', report)
    if eligible:
        winner = max(eligible, key=lambda row: (row['cv']['mean_macro_f1'], row['val']['macro_f1']))
        dump(destination / 'selection_before_repeat.json', {'candidate': winner,
            'policy': 'Highest original patient CV among candidates with nondecreasing development validation Macro-F1.',
            'external_test_evaluated': False})
    dump(out / 'state.json', {'status': 'completed', 'completed_composite_candidates': len(results),
        'eligible_for_repeat_confirmation': len(eligible), 'new_seed_fits': 0, 'external_test_evaluated': False})


if __name__ == '__main__':
    main()
