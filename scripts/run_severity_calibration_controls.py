"""Synthetic calibration experiments; freeze predictions before evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bs.severity_calibration_controls import CONTROL_METHODS as SYNTHETIC_METHODS, CalibrationControlOrderer as SyntheticDirectionOrderer


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);args=p.parse_args()
    root=ROOT/'outputs/severity_research_20261013'
    out=root/'synthetic_controls'/f'seed{args.seed}';out.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(root/'data/image_groups.csv',dtype={'dhash':str})
    original=np.load(root/'embeddings/frozen_dino_gamma.npz',allow_pickle=False)
    synthetic=np.load(root/'embeddings/frozen_dino_synthetic.npz',allow_pickle=False)
    assert np.array_equal(frame.uid.to_numpy(),original['uids'])
    assert np.array_equal(frame.uid.to_numpy(),synthetic['uids'])
    meta=json.loads((root/'embeddings/frozen_dino_synthetic.json').read_text())
    for name,digest in meta['sources_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    z=original['embeddings'];s=synthetic['embeddings'];predictions=[];responses=[];splits=[]
    for scope in ['main','near_group_stress']:
        for fold in sorted(frame.fold.unique()):
            eligible=~frame.exclude_exact_group.to_numpy()
            test=(frame.fold.to_numpy()==fold)&eligible
            train=(frame.fold.to_numpy()!=fold)&eligible
            if scope=='near_group_stress':train &= ~frame.near_group.isin(frame.loc[test,'near_group']).to_numpy()
            assert not set(frame.loc[train,'pixel_sha256'])&set(frame.loc[test,'pixel_sha256'])
            if scope=='near_group_stress':assert not set(frame.loc[train,'near_group'])&set(frame.loc[test,'near_group'])
            splits.append(dict(scope=scope,fold=fold,n_train=int(train.sum()),n_test=int(test.sum()),
                               train_uids=frame.loc[train,'uid'].tolist(),test_uids=frame.loc[test,'uid'].tolist(),
                               exact_hash_overlap=0,
                               near_group_overlap=len(set(frame.loc[train,'near_group'])&set(frame.loc[test,'near_group']))))
            for method in SYNTHETIC_METHODS:
                model=SyntheticDirectionOrderer(method,args.seed).fit(z[0,train],s[0,train],s[1,train],s[2,train],z[1,train],z[2,train])
                for gi,gamma in enumerate(original['gammas']):
                    scores,levels=model.predict(z[gi,test])
                    for uid,score,level in zip(frame.loc[test,'uid'],scores,levels):
                        predictions.append(dict(uid=uid,scope=scope,fold=fold,method=method,gamma=float(gamma),score=float(score),level=int(level)))
                raw=model.raw_score(z[0,test]);low=model.raw_score(s[0,test]);high=model.raw_score(s[1,test]);ctrl=model.raw_score(s[2,test])
                responses.append(dict(scope=scope,fold=fold,method=method,n=int(test.sum()),
                    low_gt_original_fraction=float((low>raw).mean()),high_gt_low_fraction=float((high>low).mean()),
                    monotonic_fraction=float(((low>raw)&(high>low)).mean()),
                    mean_localized_change=float((high-raw).mean()),mean_control_change=float((ctrl-raw).mean()),
                    local_gt_control_fraction=float((high>ctrl).mean())))
                states=out/'states';states.mkdir(exist_ok=True)
                joblib.dump(model,states/f'{scope}_{fold}_{method}.joblib',compress=3)
                (states/f'{scope}_{fold}_{method}.json').write_text(json.dumps(model.diagnostics,indent=2)+'\n')
                print(f'{scope} {fold} {method} heldout synthetic monotonic={responses[-1]["monotonic_fraction"]:.3f}',flush=True)
    pd.DataFrame(predictions).to_csv(out/'predictions_frozen_before_evaluation.csv',index=False)
    pd.DataFrame(responses).to_csv(out/'synthetic_dose_response.csv',index=False)
    (out/'splits.json').write_text(json.dumps(splits,indent=2)+'\n')
    sources=[Path(__file__),ROOT/'src/bs/severity_synthetic.py',ROOT/'src/bs/severity_features.py',ROOT/'docs/severity_synthetic_protocol_20261013.md',ROOT/'src/bs/severity_calibration_controls.py',ROOT/'docs/severity_calibration_controls_20261013.md']
    (out/'frozen_manifest.json').write_text(json.dumps(dict(seed=args.seed,manual_grading_labels_used=False,
        no_mask_or_clinical_label_inputs=True,synthetic_supervision=True,
        prior_real_mask_baseline_results_seen_during_method_development=True,
        predictions_sha256=hashlib.sha256((out/'predictions_frozen_before_evaluation.csv').read_bytes()).hexdigest(),
        source_sha256={str(s.relative_to(ROOT)):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources}),indent=2)+'\n')


if __name__=='__main__':main()
