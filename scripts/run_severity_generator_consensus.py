"""Freeze all three fixed aggregations and hypothesis-disagreement outputs."""
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bs.severity_generator_consensus import CONSENSUS_METHODS, GeneratorConsensusOrderer
from bs.severity_generator_sensitivity import GENERATOR_SPECS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = ROOT / 'outputs/severity_research_20261013'
    old = root / 'generator_sensitivity'
    old_manifest = json.loads((old / 'frozen_manifest.json').read_text())
    for name, expected in {**old_manifest['source_sha256'], **old_manifest['input_sha256']}.items():
        assert digest(ROOT / name) == expected, name
    old_predictions = old / 'predictions_frozen_before_evaluation.csv'
    assert digest(old_predictions) == old_manifest['predictions_sha256']
    reference = pd.read_csv(old_predictions)
    reference = reference[reference.method == 'gamma_orthogonal_direction']
    expected_states = {(a['variant'], a['scope'], a['fold'], a['method']): a['state_sha256']
                       for a in old_manifest['reload_audits']}
    variants = list(GENERATOR_SPECS)
    assert variants == ['baseline', 'narrow', 'broad', 'single', 'many', 'half_dose', 'double_dose']
    original_path = root / 'embeddings/frozen_dino_gamma.npz'
    original = np.load(original_path, allow_pickle=False)
    groups_path = root / 'registration_audit/image_registration_groups.csv'
    frame = pd.read_csv(groups_path, dtype={'dhash': str})
    assert np.array_equal(original['uids'], frame.uid.to_numpy())
    caches, inputs = {}, {str(p.relative_to(ROOT)): digest(p)
        for p in [original_path, groups_path, old_predictions, old / 'frozen_manifest.json']}
    for variant in variants:
        path = root / 'embeddings/frozen_dino_synthetic.npz' if variant == 'baseline' else old / f'{variant}.npz'
        cache = np.load(path, allow_pickle=False)
        assert np.array_equal(cache['uids'], original['uids'])
        caches[variant] = cache['embeddings']
        inputs[str(path.relative_to(ROOT))] = digest(path)
    out = root / 'generator_consensus/seed42'
    (out / 'states').mkdir(parents=True, exist_ok=True)
    predictions, disagreement, splits, audits, geometries, angles, responses = [], [], [], [], [], [], []
    z = original['embeddings']
    for scope in ['main', 'registration_group_stress']:
        for fold in ['f1', 'f2', 'f3', 'f4', 'f5']:
            eligible = ~frame.exclude_exact_group.to_numpy()
            test = eligible & (frame.fold.to_numpy() == fold)
            train = eligible & (frame.fold.to_numpy() != fold)
            if scope == 'registration_group_stress':
                train &= ~frame.registration_group.isin(frame.loc[test, 'registration_group']).to_numpy()
                assert not set(frame.loc[train, 'registration_group']) & set(frame.loc[test, 'registration_group'])
            assert not set(frame.loc[train, 'pixel_sha256']) & set(frame.loc[test, 'pixel_sha256'])
            models = []
            for variant in variants:
                path = old / 'states' / variant / f'{scope}_{fold}_gamma_orthogonal_direction.joblib'
                assert digest(path) == expected_states[(variant, scope, fold, 'gamma_orthogonal_direction')]
                inputs[str(path.relative_to(ROOT))] = digest(path)
                models.append(joblib.load(path))
            splits.append(dict(scope=scope, fold=fold, n_train=int(train.sum()), n_test=int(test.sum()),
                train_uids=frame.loc[train, 'uid'].tolist(), test_uids=frame.loc[test, 'uid'].tolist()))
            directions = np.stack([model.direction for model in models])
            gram = directions @ directions.T
            w = np.maximum(np.linalg.eigvalsh(gram), 0)
            w /= w.sum()
            entropy_rank = float(np.exp(-np.sum(w[w > 0] * np.log(w[w > 0]))))
            geometries.append(dict(scope=scope, fold=fold, n_fit=int(train.sum()),
                direction_entropy_effective_rank=entropy_rank, largest_direction_variance_fraction=float(w.max()),
                mean_direction_norm=float(np.linalg.norm(directions.mean(0))),
                min_pairwise_cosine=float(gram[np.triu_indices(len(variants), 1)].min())))
            for i, a in enumerate(variants):
                for j in range(i + 1, len(variants)):
                    cosine = float(np.clip(gram[i, j], -1, 1))
                    angles.append(dict(scope=scope, fold=fold, variant_a=a, variant_b=variants[j],
                                       cosine=cosine, angle_degrees=float(np.degrees(np.arccos(cosine)))))
            for mi, method in enumerate(CONSENSUS_METHODS):
                model = GeneratorConsensusOrderer(method).fit(z[0, train], models, variants)
                path = out / 'states' / f'{scope}_{fold}_{method}.joblib'
                joblib.dump(model, path, compress=3)
                repeat = joblib.load(path)
                max_error, reference_error = 0., 0.
                for gi, gamma in enumerate(original['gammas']):
                    scores, levels = model.predict(z[gi, test])
                    checked_scores, checked_levels = repeat.predict(z[gi, test])
                    error = float(np.max(np.abs(scores - checked_scores)))
                    max_error = max(max_error, error)
                    assert error < 1e-12 and np.array_equal(levels, checked_levels)
                    predictions.extend(dict(scope=scope, fold=fold, method=method, gamma=float(gamma),
                        uid=uid, score=float(score), level=int(level))
                        for uid, score, level in zip(frame.loc[test, 'uid'], scores, levels))
                    if mi == 0:
                        detail = repeat.disagreement(z[gi, test])
                        for vi, variant in enumerate(variants):
                            previous = reference[(reference.variant == variant) & (reference.scope == scope) &
                                (reference.fold == fold) & (reference.gamma == gamma)].set_index('uid').loc[frame.loc[test, 'uid']]
                            difference = float(np.max(np.abs(detail['generator_scores'][:, vi] - previous.score)))
                            reference_error = max(reference_error, difference)
                            assert difference < 1e-12 and np.array_equal(detail['generator_levels'][:, vi], previous.level)
                        for i, uid in enumerate(frame.loc[test, 'uid']):
                            record = dict(scope=scope, fold=fold, gamma=float(gamma), uid=uid)
                            for key in ['rank_min', 'rank_max', 'rank_range']:
                                record[key] = float(detail[key][i])
                            for key in ['level_min', 'level_max', 'common_level']:
                                record[key] = int(detail[key][i])
                            record['all_generators_agree_level'] = bool(detail['all_generators_agree_level'][i])
                            for vi, variant in enumerate(variants):
                                record[f'{variant}_score'] = float(detail['generator_scores'][i, vi])
                                record[f'{variant}_level'] = int(detail['generator_levels'][i, vi])
                            disagreement.append(record)
                audits.append(dict(scope=scope, fold=fold, state_sha256=digest(path),
                    all_test_scores_reloaded=True, max_reload_error=max_error,
                    all_input_generator_test_scores_replayed=mi == 0, max_reference_error=reference_error,
                    **model.diagnostics))
                raw = repeat.raw_score(z[0, test])
                for variant, cache in caches.items():
                    low, high = repeat.raw_score(cache[0, test]), repeat.raw_score(cache[1, test])
                    responses.append(dict(scope=scope, fold=fold, method=method, variant=variant, n=int(test.sum()),
                        low_gt_original_fraction=float((low > raw).mean()),
                        high_gt_low_fraction=float((high > low).mean()),
                        strict_two_step_monotonic_fraction=float(((low > raw) & (high > low)).mean())))
            print(scope, fold, 'three consensus states frozen', flush=True)
    prediction = pd.DataFrame(predictions)
    detail = pd.DataFrame(disagreement)
    assert len(prediction) == 36990 and len(detail) == 12330
    assert prediction.uid.nunique() == 2055 and len(audits) == 30
    assert not prediction.duplicated(['scope', 'uid', 'gamma', 'method']).any()
    prediction_path = out / 'predictions_frozen_before_evaluation.csv'
    prediction.to_csv(prediction_path, index=False)
    detail_path = out / 'generator_disagreement_frozen_before_evaluation.csv'
    detail.to_csv(detail_path, index=False)
    pd.DataFrame(geometries).to_csv(out / 'fit_direction_geometry.csv', index=False)
    pd.DataFrame(angles).to_csv(out / 'fit_pairwise_direction_angles.csv', index=False)
    pd.DataFrame(responses).to_csv(out / 'heldout_synthetic_dose_response.csv', index=False)
    (out / 'splits.json').write_text(json.dumps(splits, indent=2) + '\n')
    sources = [Path(__file__), ROOT / 'src/bs/severity_generator_consensus.py',
               ROOT / 'docs/severity_generator_consensus_protocol_20261013.md']
    manifest = dict(seed=42, states=30, n_predictions=len(prediction),
        predictions_sha256=digest(prediction_path), disagreement_sha256=digest(detail_path),
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources}, input_sha256=inputs,
        reload_audits=audits, generator_variants=variants, equal_weights=True,
        fitting_reads_no_masks_or_clinical_labels=True, previous_proxy_results_known=True,
        original_generator_remains_primary=True, clinical_grading_validated=False, patient_independence_proven=False)
    (out / 'frozen_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('All 30 consensus states and 36990 predictions frozen; masks were not opened', flush=True)


if __name__ == '__main__':
    main()
