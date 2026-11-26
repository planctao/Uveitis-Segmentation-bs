"""Replay every frozen saliency rank output and verify archived source evidence."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'outputs/severity_research_20261013'
sys.path.insert(0, str(ROOT/'src'))
from bs.severity_saliency import METHODS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(hashes):
    for name, expected in hashes.items():
        assert digest(ROOT/name) == expected, name


def main():
    directory = RUN/'classical_leakage_baseline'
    feature_path, frozen_path = directory/'feature_manifest.json', directory/'frozen_manifest.json'
    feature = json.loads(feature_path.read_text())
    frozen = json.loads(frozen_path.read_text())
    for record in [feature, frozen]:
        verify(record['source_sha256']); verify(record['input_sha256'])
    assert digest(feature_path) == frozen['feature_manifest_sha256']
    assert feature['n_images'] == frozen['n_images'] == 2055 and not feature['probe']
    raw_path = directory/'raw_scores.csv'
    pred_path = directory/'predictions_frozen_before_evaluation.csv'
    assert digest(raw_path) == feature['raw_scores_sha256'] == frozen['raw_scores_sha256']
    assert digest(pred_path) == frozen['predictions_sha256']
    assert digest(directory/'splits.json') == frozen['splits_sha256']
    raw = pd.read_csv(raw_path, float_precision='round_trip')
    pred = pd.read_csv(pred_path, float_precision='round_trip')
    assert len(raw) == len(pred) == frozen['n_predictions'] == 24660
    assert not raw.duplicated(['uid', 'method', 'condition']).any()
    assert not pred.duplicated(['uid', 'method', 'condition']).any()
    assert pred.groupby(['method','condition']).size().eq(2055).all()
    assert set(pred.method) == set(METHODS) and set(pred.condition) == set(frozen['conditions'])
    assert pred.score.between(0, 1).all() and pred.level.isin([0, 1, 2, 3]).all()
    images = pd.read_csv(RUN/'data/image_groups.csv', dtype={'dhash':str})
    eligible = images[~images.exclude_exact_group].set_index('uid')
    assert set(pred.uid) == set(eligible.index) == set(feature['raw_shards_sha256'])
    shard_checks = []
    for uid, expected in feature['raw_shards_sha256'].items():
        path = directory/'raw'/f'{uid}.npz'
        assert digest(path) == expected
        with np.load(path, allow_pickle=False) as stored:
            assert stored['fingerprint'].item() == feature['fingerprint']
            assert stored['pixel_sha256'].item() == eligible.loc[uid,'pixel_sha256']
            actual = raw[raw.uid == uid].set_index(['condition','method']).loc[
                [(c,m) for c in frozen['conditions'] for m in METHODS], 'raw_score'].to_numpy().reshape(3, 4)
            error = float(np.max(np.abs(actual-stored['scores'])))
            assert error < 1e-15
            shard_checks.append(error)
    splits = json.loads((directory/'splits.json').read_text())
    split_by_fold = {s['fold']:s for s in splits}
    maximum_rank_error, n_replayed = 0., 0
    for record in frozen['reload_audits']:
        assert digest(ROOT/record['state_path']) == record['state_sha256']
        state = joblib.load(ROOT/record['state_path'])
        split = split_by_fold[record['fold']]
        assert not set(split['train_uids'])&set(split['test_uids'])
        train = raw[(raw.method == record['method'])&(raw.condition == 'identity')].set_index('uid').loc[split['train_uids']]
        np.testing.assert_allclose(state.sorted_scores, np.sort(train.raw_score), atol=1e-15, rtol=0)
        np.testing.assert_allclose(state.cutpoints, np.quantile(train.raw_score,[.25,.5,.75]), atol=1e-15, rtol=0)
        for condition in frozen['conditions']:
            part = pred[(pred.fold == record['fold'])&(pred.method == record['method'])&
                (pred.condition == condition)].set_index('uid').loc[split['test_uids']]
            scores, levels = state.predict(part.raw_score.to_numpy())
            error = float(np.max(np.abs(scores-part.score.to_numpy())))
            assert error < 1e-15 and np.array_equal(levels, part.level.to_numpy())
            maximum_rank_error = max(maximum_rank_error, error); n_replayed += len(part)
    assert n_replayed == 24660
    reading_path = directory/'literature/reading_audit.json'
    reading = json.loads(reading_path.read_text())
    for field in ['sources_sha256','retrieval_manifest_sha256','archived_response_sha256']:
        verify(reading[field])
    assert reading['author_manuscript_full_text_read'] and reading['formula_pages_visually_reviewed'] == [4]
    evaluation_path = directory/'evaluation/audit.json'
    evaluation = json.loads(evaluation_path.read_text())
    verify(evaluation['source_sha256']); verify(evaluation['input_sha256'])
    for name, expected in evaluation['output_sha256'].items():
        assert digest(directory/'evaluation'/name) == expected, name
    assert evaluation['original_primary_metrics_reproduced']
    assert evaluation['all_conditions_frozen_before_this_target_read']
    assert evaluation['posthoc_development_comparison']
    assert len(pd.read_csv(directory/'evaluation/pooled_metrics.csv')) == evaluation['n_pooled_rows'] == 18
    assert len(pd.read_csv(directory/'evaluation/fold_metrics.csv')) == evaluation['n_fold_rows'] == 90
    illustration_path = directory/'illustrations/audit.json'
    illustrations = json.loads(illustration_path.read_text())
    verify(illustrations['source_sha256']); verify(illustrations['input_sha256'])
    for name, expected in illustrations['output_sha256'].items():
        assert digest(directory/'illustrations'/name) == expected
    for record in illustrations['records']:
        part = raw[(raw.uid == record['uid'])&(raw.condition == 'identity')].set_index('method').loc[list(METHODS)]
        np.testing.assert_allclose(part.raw_score, record['scores'], rtol=0, atol=1e-15)
    visual_path = directory/'visual_review.json'
    visual = json.loads(visual_path.read_text())
    verify(visual['input_sha256'])
    assert visual['all_three_images_actually_viewed'] and len(visual['input_sha256']) == 3
    assert not visual['english_manuscript_document_render_review_completed']
    tests_path = directory/'unit_tests.log'
    counts = re.findall(r'(\d+) passed in', tests_path.read_text())
    assert counts and int(counts[-1]) == 4
    clinical_path = RUN/'clinical_evaluation/model_lock_manifest.json'
    assert digest(clinical_path) == frozen['original_clinical_lock_sha256']
    clinical = json.loads(clinical_path.read_text())
    verify(clinical['source_sha256']); verify(clinical['input_sha256'])
    for field in ['primary_model_changed','clinical_lock_changed','clinical_grading_validated',
                  'patient_independence_proven','external_validation_performed','goal_achieved',
                  'full_published_leakage_segmenter_reproduced']:
        assert not evaluation[field]
    sources = [Path(__file__), ROOT/'tests/test_severity_saliency.py',
               ROOT/'docs/severity_saliency_findings_20261013.md']
    report = dict(updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},
        input_sha256={str(p.relative_to(ROOT)):digest(p) for p in
            [feature_path, frozen_path, raw_path, pred_path, reading_path, evaluation_path, illustration_path, visual_path, tests_path]},
        n_images=2055, n_states=20, n_raw_shards_verified=2055,
        n_predictions_replayed=n_replayed, maximum_rank_error=maximum_rank_error,
        maximum_raw_csv_error=max(shard_checks), n_pooled_rows=18, n_fold_rows=90,
        tests_passed=4, test_log_sha256=digest(tests_path),
        all_sources_and_archived_responses_verified=True,
        all_raw_shards_and_prediction_arrays_verified=True,
        posthoc_development_ranking_adaptation=True,
        full_published_leakage_segmenter_reproduced=False,
        primary_model_changed=False, clinical_lock_changed=False,
        clinical_grading_validated=False, patient_independence_proven=False,
        external_validation_performed=False, dataset_modified=False, goal_achieved=False)
    (directory/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['n_images','n_states','n_raw_shards_verified',
        'n_predictions_replayed','maximum_rank_error','n_pooled_rows','n_fold_rows','tests_passed','goal_achieved']}, indent=2))


if __name__ == '__main__':
    main()
