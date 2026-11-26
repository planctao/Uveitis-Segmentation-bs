"""Replay deployment on local caches; report no external/clinical performance."""
import datetime
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bs.severity_transfer import digest, load_bundle, predict_bundle
from run_severity_external_inference import extract_images

RUN = ROOT/'outputs/severity_research_20261013'


def main():
    out = RUN/'external_transfer'
    lock_path = out/'model_lock_manifest.json'
    lock, groups = load_bundle(ROOT,lock_path)
    feature_path = RUN/'embeddings/frozen_dino_gamma.npz'
    data = np.load(feature_path,allow_pickle=False)
    frame = pd.read_csv(RUN/'data/image_groups.csv',dtype={'dhash':str})
    assert np.array_equal(data['uids'],frame.uid.to_numpy())
    keep = ~frame.exclude_exact_group.to_numpy()
    selected = frame[keep].copy()
    fold_outputs, aggregate_outputs = [], []
    inputs = {str(feature_path.relative_to(ROOT)):digest(feature_path),
        str(lock_path.relative_to(ROOT)):digest(lock_path)}
    references = {}
    for name in lock['methods']:
        path = RUN/('baselines/seed42' if name == 'mean_intensity' else 'registration_stress/seed42')/'predictions_frozen_before_evaluation.csv'
        inputs[str(path.relative_to(ROOT))] = digest(path)
        references[name] = pd.read_csv(path,float_precision='round_trip')
    replayed, max_error, detailed = 0, 0., []
    for gi,gamma in enumerate(data['gammas']):
        folds, aggregate = predict_bundle(groups,selected.uid.to_numpy(),data['embeddings'][gi,keep],data['image_features'][gi,keep])
        folds['gamma'], aggregate['gamma'] = gamma,gamma
        fold_outputs.append(folds); aggregate_outputs.append(aggregate)
        heldout = folds.merge(selected[['uid','fold']],on=['uid','fold'],validate='many_to_one')
        for name in lock['methods']:
            method = next(record['state_method'] for record in lock['states'] if record['name'] == name)
            reference = references[name]
            reference = reference[(reference.method == method)&(reference.gamma == gamma)]
            if name == 'mean_intensity':
                reference = reference[reference.scope == 'main']
            joined = heldout[heldout.method == name].merge(reference,on=['uid','fold'],validate='one_to_one',suffixes=('_replayed','_archived'))
            assert len(joined) == 2055
            error = float(np.max(np.abs(joined.score_replayed-joined.score_archived)))
            assert error < 1e-15 and np.array_equal(joined.local_relative_level,joined.level)
            replayed += len(joined); max_error = max(max_error,error)
            detailed.append(dict(method=name,gamma=float(gamma),n=len(joined),maximum_cdf_error=error,levels_identical=True))
    folds = pd.concat(fold_outputs,ignore_index=True)
    aggregate = pd.concat(aggregate_outputs,ignore_index=True)
    assert len(folds) == 154125 and len(aggregate) == replayed == 30825
    grouped = folds.groupby(['uid','method','gamma']).score.mean()
    joined = aggregate.set_index(['uid','method','gamma'])
    np.testing.assert_allclose(joined.score,grouped.loc[joined.index],rtol=0,atol=1e-15)
    folds.to_csv(out/'local_implementation_fold_outputs.csv',index=False)
    aggregate.to_csv(out/'local_implementation_aggregate_outputs.csv',index=False)
    # Check original-image extraction on fixed source indices, not chosen by scores.
    images = frame.iloc[:8][['uid','image_path']].copy()
    z, f, image_audits = extract_images(images,ROOT/lock['encoder_weights'],'cuda',8)
    feature_error = float(np.max(np.abs(z-data['embeddings'][:,:8])))
    descriptor_error = float(np.max(np.abs(f-data['image_features'][:,:8])))
    assert feature_error <= .005 and descriptor_error == 0, (feature_error,descriptor_error)
    image_replays, pixel_checks = [], []
    for i, record in enumerate(image_audits):
        assert record['pixel_sha256'] == frame.iloc[i].pixel_sha256
        pixel_checks.append(record)
    max_cdf_error, changed_levels = 0., 0
    for gi, gamma in enumerate(data['gammas']):
        current,_ = predict_bundle(groups,images.uid.to_numpy(),z[gi],f[gi])
        cached,_ = predict_bundle(groups,images.uid.to_numpy(),data['embeddings'][gi,:8],data['image_features'][gi,:8])
        np.testing.assert_array_equal(current[['uid','method','fold']],cached[['uid','method','fold']])
        error = float(np.max(np.abs(current.score-cached.score)))
        max_cdf_error = max(max_cdf_error,error)
        changed_levels += int((current.local_relative_level != cached.local_relative_level).sum())
        current['gamma'] = gamma; image_replays.append(current)
    assert max_cdf_error <= .003 and changed_levels == 0, (max_cdf_error,changed_levels)
    pd.concat(image_replays).to_csv(out/'local_original_image_replay.csv',index=False)
    (out/'local_image_provenance.json').write_text(json.dumps(pixel_checks,indent=2)+'\n')
    test_log = out/'unit_tests.log'
    with test_log.open('w') as stream:
        subprocess.run([sys.executable,'-m','pytest','tests/test_severity_transfer.py','-q'],cwd=ROOT,
            stdout=stream,stderr=subprocess.STDOUT,check=True)
    import re
    assert re.search(r'5 passed in',test_log.read_text())
    access = out/'access_probe'
    access_paths = sorted(p for p in access.iterdir() if p.is_file())
    official = access/'duke_official_page.html'
    assert digest(official) == digest(RUN/'external_data_search/duke_rabbani_dataset_verified.html')
    clinical_path = RUN/'clinical_evaluation/model_lock_manifest.json'
    assert digest(clinical_path) == lock['clinical_lock_sha256']
    artifacts = [out/'local_implementation_fold_outputs.csv',out/'local_implementation_aggregate_outputs.csv',
        out/'local_original_image_replay.csv',out/'local_image_provenance.json',test_log]
    report = dict(updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=lock['source_sha256'],input_sha256=inputs,
        access_evidence_sha256={str(p.relative_to(ROOT)):digest(p) for p in access_paths},
        output_sha256={p.name:digest(p) for p in artifacts},
        n_states=25,n_local_images=2055,n_local_fold_outputs=len(folds),
        n_local_aggregate_outputs=len(aggregate),n_heldout_predictions_replayed=replayed,
        maximum_heldout_cdf_error=max_error,heldout_levels_identical=True,replay_checks=detailed,
        n_original_images_reextracted=8,n_original_image_fold_outputs=600,
        maximum_reextracted_embedding_error=feature_error,maximum_reextracted_descriptor_error=descriptor_error,
        maximum_original_image_cdf_error=max_cdf_error,original_image_level_changes=changed_levels,
        tests_passed=5,implementation_uses_fixed_local_CDFs=True,
        local_five_fold_aggregate_is_not_heldout_performance=True,
        local_original_image_replay_is_not_external_validation=True,
        raw_external_images_obtained=0,external_targets_read=False,
        external_validation_performed=False,clinical_grading_validated=False,
        model_refitted=False,clinical_primary_model_changed=False,
        external_cohort_lock_created=False,patient_independence_proven=False,
        dataset_modified=False,external_messages_sent=False,goal_achieved=False)
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['n_states','n_local_images','n_local_fold_outputs',
        'n_heldout_predictions_replayed','maximum_heldout_cdf_error','n_original_images_reextracted',
        'maximum_reextracted_embedding_error','maximum_original_image_cdf_error','tests_passed',
        'external_validation_performed']},indent=2))


if __name__ == '__main__':
    main()
