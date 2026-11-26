"""Fit image-only C4 controls, replay every frozen score, then seal outputs."""
import datetime
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bs.severity_geometric_stress import DIHEDRAL_CONDITIONS
from bs.severity_rotation_pooling import RotationFeatureMeanOrderer, RotationScoreMeanOrderer
from bs.severity_synthetic import SYNTHETIC_METHODS

RUN = ROOT/'outputs/severity_research_20261013'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = RUN/'rotation_pooling'
    assert not (out/'frozen_manifest.json').exists(), 'Completed predictions must not be overwritten'
    metadata_path = out/'feature_manifest.json'
    metadata = json.loads(metadata_path.read_text())
    for key in ['source_sha256','input_sha256']:
        for name,expected in metadata[key].items():assert digest(ROOT/name)==expected,name
    feature_path = out/'frozen_rotation_orbits.npz'
    assert digest(feature_path)==metadata['features_sha256']
    geometry_path = RUN/'geometric_stress/frozen_manifest.json'
    geometry = json.loads(geometry_path.read_text())
    for key in ['source_sha256','input_sha256']:
        for name,expected in geometry[key].items():assert digest(ROOT/name)==expected,name
    frame_path = RUN/'data/image_groups.csv'
    frame = pd.read_csv(frame_path,dtype={'dhash':str})
    features = np.load(feature_path,allow_pickle=False)
    assert np.array_equal(features['uids'],frame.uid.to_numpy())
    original,synthetic = features['original_orbits'],features['synthetic_orbits']
    assert original.shape==synthetic.shape==(3,4,2189,768)
    conditions = {}
    for k,condition in enumerate(DIHEDRAL_CONDITIONS[:4]):
        conditions[condition] = np.roll(original[0],-k,axis=0)
    for k,condition in enumerate(DIHEDRAL_CONDITIONS[4:]):
        conditions[condition] = np.roll(features['reflected_orbit'],-k,axis=0)
    conditions.update(gamma08=original[1],gamma12=original[2])
    inputs = {str(p.relative_to(ROOT)):digest(p) for p in [
        metadata_path,feature_path,geometry_path,frame_path,RUN/'synthetic_calibration/seed42/splits.json']}
    old_splits = json.loads((RUN/'synthetic_calibration/seed42/splits.json').read_text())
    combinations = [('rotation_feature_mean',seed) for seed in [42,2026,3407]] + [('rotation_score_mean',42)]
    states_dir = out/'states';states_dir.mkdir(exist_ok=True)
    predictions,responses,audits,splits = [],[],[],[]
    for fold in sorted(frame.fold.unique()):
        eligible = ~frame.exclude_exact_group.to_numpy()
        test = eligible & (frame.fold.to_numpy()==fold)
        train = eligible & (frame.fold.to_numpy()!=fold)
        assert not set(frame.loc[train,'pixel_sha256'])&set(frame.loc[test,'pixel_sha256'])
        old = next(s for s in old_splits if s['scope']=='main' and s['fold']==fold)
        assert set(old['train_uids'])==set(frame.loc[train,'uid'])
        assert set(old['test_uids'])==set(frame.loc[test,'uid'])
        splits.append(dict(scope='main',fold=fold,n_train=int(train.sum()),n_test=int(test.sum()),
            train_uids=frame.loc[train,'uid'].tolist(),test_uids=frame.loc[test,'uid'].tolist()))
        for branch,seed in combinations:
            for method in SYNTHETIC_METHODS:
                if branch=='rotation_feature_mean':
                    fit_orbits = [original[0],*synthetic,original[1],original[2]]
                    model = RotationFeatureMeanOrderer(method,seed).fit(*[orbit[:,train,:] for orbit in fit_orbits])
                else:
                    path = RUN/f'synthetic_calibration/seed42/states/main_{fold}_{method}.joblib'
                    assert digest(path)==geometry['input_sha256'][str(path.relative_to(ROOT))]
                    inputs[str(path.relative_to(ROOT))] = digest(path)
                    base = joblib.load(path)
                    assert base.diagnostics['n_fit']==int(train.sum())
                    model = RotationScoreMeanOrderer().fit(original[0][:,train,:],base)
                assert model.diagnostics['n_fit']==int(train.sum())
                state_path = states_dir/f'{branch}_seed{seed}_{fold}_{method}.joblib'
                joblib.dump(model,state_path,compress=3)
                reloaded = joblib.load(state_path)
                max_score_error,max_raw_error = 0.,0.
                identity_scores = identity_levels = None
                for condition,orbit in conditions.items():
                    heldout = orbit[:,test,:]
                    score,level = model.predict(heldout)
                    raw = model.raw_score(heldout)
                    replay_score,replay_level = reloaded.predict(heldout)
                    replay_raw = reloaded.raw_score(heldout)
                    score_error = float(np.max(np.abs(score-replay_score)))
                    raw_error = float(np.max(np.abs(raw-replay_raw)))
                    max_score_error = max(max_score_error,score_error)
                    max_raw_error = max(max_raw_error,raw_error)
                    assert score_error<1e-12 and raw_error<1e-12 and np.array_equal(level,replay_level)
                    if condition=='identity':
                        identity_scores,identity_levels = score,level
                    if condition in DIHEDRAL_CONDITIONS[1:4]:
                        assert np.array_equal(score,identity_scores) and np.array_equal(level,identity_levels)
                    predictions.extend(dict(uid=uid,scope='main',fold=fold,branch=branch,seed=seed,method=method,
                        condition=condition,score=float(s),raw_score=float(r),level=int(l))
                        for uid,s,r,l in zip(frame.loc[test,'uid'],score,raw,level))
                for part,selected in [('fit',train),('heldout',test)]:
                    raw = model.raw_score(original[0][:,selected,:])
                    low,high,control = [model.raw_score(s[:,selected,:]) for s in synthetic]
                    responses.append(dict(branch=branch,seed=seed,fold=fold,method=method,split=part,n=int(selected.sum()),
                        strict_two_step_monotonic_fraction=float(((raw<low)&(low<high)).mean()),
                        high_gt_control_fraction=float((high>control).mean()),
                        mean_localized_change=float((high-raw).mean()),mean_control_change=float((control-raw).mean())))
                audits.append(dict(branch=branch,seed=seed,fold=fold,method=method,n_fit=int(train.sum()),
                    n_test=int(test.sum()),state_path=str(state_path.relative_to(ROOT)),state_sha256=digest(state_path),
                    all_test_conditions_reloaded=True,max_score_error=max_score_error,max_raw_error=max_raw_error,
                    c4_scores_and_levels_bitwise_equal=True,masks_used=False,clinical_labels_used=False))
                print(branch,seed,fold,method,'frozen and all conditions replayed',flush=True)
    pred = pd.DataFrame(predictions)
    assert len(audits)==80 and len(pred)==328800 and pred.uid.nunique()==2055
    assert not pred.duplicated(['branch','seed','uid','method','condition']).any()
    assert pred.level.isin([0,1,2,3]).all() and pred.score.between(0,1).all()
    assert np.isfinite(pred[['score','raw_score']]).all().all()
    prediction_path = out/'predictions_frozen_before_evaluation.csv'
    pred.to_csv(prediction_path,index=False)
    response_path = out/'synthetic_dose_response.csv'
    pd.DataFrame(responses).to_csv(response_path,index=False)
    split_path = out/'splits.json';split_path.write_text(json.dumps(splits,indent=2)+'\n')
    sources = [Path(__file__),ROOT/'src/bs/severity_rotation_pooling.py',ROOT/'src/bs/severity_synthetic.py',
        ROOT/'src/bs/severity_geometric_stress.py',ROOT/'src/bs/severity_features.py',
        ROOT/'docs/severity_rotation_pooling_protocol_20261013.md']
    for name,expected in inputs.items():assert digest(ROOT/name)==expected,name
    result = dict(frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),states=80,
        n_predictions=len(pred),n_images=2055,conditions=list(conditions),branches_and_seeds=combinations,
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},input_sha256=inputs,
        predictions_sha256=digest(prediction_path),synthetic_response_sha256=digest(response_path),
        splits_sha256=digest(split_path),reload_audits=audits,
        no_mask_or_clinical_label_inputs=True,neural_weights_trained=False,
        prior_proxy_results_seen_during_development=True,original_clinical_lock_preserved=True,
        primary_model_changed=False,clinical_grading_validated=False,patient_independence_proven=False)
    (out/'frozen_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print('C4 frozen predictions and reload audit complete:',len(pred),'predictions',len(audits),'states',flush=True)


if __name__ == '__main__':
    main()
