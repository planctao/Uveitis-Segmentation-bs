"""Refit frozen-DINO scorers after excluding held-out registration groups."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bs.severity_synthetic import SYNTHETIC_METHODS, SyntheticDirectionOrderer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, choices=[42, 2026, 3407])
    args = parser.parse_args()
    root = ROOT / 'outputs/severity_research_20261013'
    registration = root / 'registration_audit'
    audit = json.loads((registration / 'audit.json').read_text())
    assert audit['masks_read'] is False
    assert digest(ROOT / 'scripts/audit_severity_registration.py') == audit['script_sha256']
    assert digest(ROOT / 'docs/severity_registration_protocol_20261013.md') == audit['protocol_sha256']
    frame = pd.read_csv(registration / 'image_registration_groups.csv', dtype={'dhash': str})
    original = np.load(root / 'embeddings/frozen_dino_gamma.npz', allow_pickle=False)
    synthetic = np.load(root / 'embeddings/frozen_dino_synthetic.npz', allow_pickle=False)
    assert np.array_equal(frame.uid, original['uids']) and np.array_equal(frame.uid, synthetic['uids'])
    metadata = json.loads((root / 'embeddings/frozen_dino_synthetic.json').read_text())
    for name, expected in metadata['sources_sha256'].items():
        assert digest(ROOT / name) == expected, name
    out = root / 'registration_stress' / f'seed{args.seed}'
    states = out / 'states'
    states.mkdir(parents=True, exist_ok=True)
    predictions, splits, reloads, responses = [], [], [], []
    z, s = original['embeddings'], synthetic['embeddings']
    eligible = ~frame.exclude_exact_group.to_numpy()
    for fold in sorted(frame.fold.unique()):
        test = (frame.fold.to_numpy() == fold) & eligible
        main_train = (frame.fold.to_numpy() != fold) & eligible
        train = main_train & ~frame.registration_group.isin(frame.loc[test, 'registration_group']).to_numpy()
        assert not set(frame.loc[train, 'registration_group']) & set(frame.loc[test, 'registration_group'])
        assert not set(frame.loc[train, 'pixel_sha256']) & set(frame.loc[test, 'pixel_sha256'])
        splits.append(dict(scope='registration_group_stress', fold=fold,
            n_main_train=int(main_train.sum()), n_train=int(train.sum()), n_test=int(test.sum()),
            n_additionally_excluded=int((main_train & ~train).sum()),
            train_uids=frame.loc[train, 'uid'].tolist(), test_uids=frame.loc[test, 'uid'].tolist(),
            additionally_excluded_uids=frame.loc[main_train & ~train, 'uid'].tolist(),
            exact_hash_overlap=0, registration_group_overlap=0))
        for method in SYNTHETIC_METHODS:
            model = SyntheticDirectionOrderer(method, args.seed).fit(z[0, train], s[0, train], s[1, train], s[2, train], z[1, train], z[2, train])
            path = states / f'{fold}_{method}.joblib'
            joblib.dump(model, path, compress=3)
            reloaded = joblib.load(path)
            max_error = 0.
            for gi, gamma in enumerate(original['gammas']):
                scores, levels = model.predict(z[gi, test])
                recomputed, new_levels = reloaded.predict(z[gi, test])
                max_error = max(max_error, float(np.max(np.abs(scores - recomputed))))
                assert np.allclose(scores, recomputed, atol=1e-12, rtol=0) and np.array_equal(levels, new_levels)
                predictions.extend(dict(uid=uid, scope='registration_group_stress', fold=fold,
                    method=method, gamma=float(gamma), score=float(score), level=int(level))
                    for uid, score, level in zip(frame.loc[test, 'uid'], scores, levels))
            raw, low, high, control = [model.raw_score(a[test]) for a in [z[0], s[0], s[1], s[2]]]
            responses.append(dict(fold=fold, method=method, n=int(test.sum()),
                monotonic_fraction=float(((low > raw) & (high > low)).mean()),
                local_gt_control_fraction=float((high > control).mean())))
            reloads.append(dict(fold=fold, max_reloaded_prediction_difference=max_error,
                state_sha256=digest(path), **model.diagnostics))
            print(f'{args.seed} {fold} {method} n_fit={train.sum()} excluded={main_train.sum()-train.sum()}', flush=True)
    path = out / 'predictions_frozen_before_evaluation.csv'
    pred = pd.DataFrame(predictions)
    assert pred.uid.nunique() == 2055 and len(pred) == 2055 * 4 * 3
    assert not pred.duplicated(['uid', 'gamma', 'method']).any()
    pred.to_csv(path, index=False)
    pd.DataFrame(responses).to_csv(out / 'synthetic_dose_response.csv', index=False)
    (out / 'splits.json').write_text(json.dumps(splits, indent=2) + '\n')
    sources = [Path(__file__), ROOT / 'src/bs/severity_synthetic.py', ROOT / 'docs/severity_registration_protocol_20261013.md', ROOT / 'docs/severity_registration_execution_note_20261013.md']
    inputs = [registration / 'image_registration_groups.csv', registration / 'audit.json',
        root / 'embeddings/frozen_dino_gamma.npz', root / 'embeddings/frozen_dino_synthetic.npz']
    (out / 'frozen_manifest.json').write_text(json.dumps(dict(seed=args.seed,
        predictions_sha256=digest(path), source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
        input_sha256={str(p.relative_to(ROOT)): digest(p) for p in inputs},
        no_mask_or_clinical_label_inputs=True, models_refit_after_exclusion=True,
        clinical_grading_validated=False, patient_independence_proven=False, reload_audits=reloads), indent=2) + '\n')


if __name__ == '__main__':
    main()
