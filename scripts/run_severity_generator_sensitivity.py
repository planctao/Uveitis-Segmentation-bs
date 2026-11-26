"""Freeze every predeclared variant before evaluating any new mask outcome."""
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from bs.severity_synthetic import SYNTHETIC_METHODS, SyntheticDirectionOrderer
from bs.severity_generator_sensitivity import GENERATOR_SPECS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = ROOT/'outputs/severity_research_20261013'
    out = root/'generator_sensitivity'
    complete = json.loads((out/'extraction_complete.json').read_text())
    assert complete['status'] == 'complete'
    for name, expected in complete['sources_sha256'].items():
        assert digest(ROOT/name) == expected, name
    frame = pd.read_csv(root/'registration_audit/image_registration_groups.csv',dtype={'dhash':str})
    original = np.load(root/'embeddings/frozen_dino_gamma.npz',allow_pickle=False)
    assert np.array_equal(frame.uid,original['uids'])
    z = original['embeddings']
    pred, responses, splits, audits, inputs = [], [], [], [], {}
    input_paths = [root/'registration_audit/image_registration_groups.csv',root/'embeddings/frozen_dino_gamma.npz']
    for variant in GENERATOR_SPECS:
        path = root/'embeddings/frozen_dino_synthetic.npz' if variant == 'baseline' else out/f'{variant}.npz'
        input_paths.append(path)
        if variant != 'baseline':
            metadata = json.loads((out/f'{variant}.json').read_text())
            assert digest(path) == metadata['cache_sha256']
        cache = np.load(path,allow_pickle=False)
        assert np.array_equal(frame.uid,cache['uids'])
        s = cache['embeddings']
        for scope in ['main','registration_group_stress']:
            for fold in sorted(frame.fold.unique()):
                eligible = ~frame.exclude_exact_group.to_numpy()
                test = (frame.fold.to_numpy() == fold) & eligible
                train = (frame.fold.to_numpy() != fold) & eligible
                if scope == 'registration_group_stress':
                    train &= ~frame.registration_group.isin(frame.loc[test,'registration_group']).to_numpy()
                    assert not set(frame.loc[train,'registration_group']) & set(frame.loc[test,'registration_group'])
                assert not set(frame.loc[train,'pixel_sha256']) & set(frame.loc[test,'pixel_sha256'])
                splits.append(dict(variant=variant,scope=scope,fold=fold,n_train=int(train.sum()),n_test=int(test.sum()),
                    train_uids=frame.loc[train,'uid'].tolist(),test_uids=frame.loc[test,'uid'].tolist()))
                states = out/'states'/variant
                states.mkdir(parents=True,exist_ok=True)
                for method in SYNTHETIC_METHODS:
                    model = SyntheticDirectionOrderer(method,42).fit(z[0,train],s[0,train],s[1,train],s[2,train],z[1,train],z[2,train])
                    state_path = states/f'{scope}_{fold}_{method}.joblib'
                    joblib.dump(model,state_path,compress=3)
                    reloaded = joblib.load(state_path)
                    max_difference = 0.
                    for gi,gamma in enumerate(original['gammas']):
                        score,level = model.predict(z[gi,test])
                        repeat,new_level = reloaded.predict(z[gi,test])
                        max_difference = max(max_difference,float(np.max(np.abs(score-repeat))))
                        assert np.allclose(score,repeat,atol=1e-12,rtol=0) and np.array_equal(level,new_level)
                        pred.extend(dict(variant=variant,scope=scope,fold=fold,method=method,gamma=float(gamma),uid=uid,score=float(a),level=int(b))
                            for uid,a,b in zip(frame.loc[test,'uid'],score,level))
                    raw,low,high,control = [model.raw_score(a[test]) for a in [z[0],s[0],s[1],s[2]]]
                    responses.append(dict(variant=variant,scope=scope,fold=fold,method=method,n=int(test.sum()),
                        monotonic_fraction=float(((low>raw)&(high>low)).mean()),
                        local_gt_control_fraction=float((high>control).mean())))
                    audits.append(dict(variant=variant,scope=scope,fold=fold,state_sha256=digest(state_path),
                        max_reload_difference=max_difference,**model.diagnostics))
                print(variant,scope,fold,'frozen',flush=True)
    predictions = pd.DataFrame(pred)
    assert len(predictions) == 7*2*4*3*2055
    assert not predictions.duplicated(['variant','scope','uid','method','gamma']).any()
    path = out/'predictions_frozen_before_evaluation.csv'
    predictions.to_csv(path,index=False)
    pd.DataFrame(responses).to_csv(out/'synthetic_dose_response.csv',index=False)
    (out/'splits.json').write_text(json.dumps(splits,indent=2)+'\n')
    sources = [Path(__file__),ROOT/'src/bs/severity_synthetic.py',ROOT/'src/bs/severity_generator_sensitivity.py',
        ROOT/'docs/severity_generator_sensitivity_protocol_20261013.md']
    (out/'frozen_manifest.json').write_text(json.dumps(dict(seed=42,states=280,n_predictions=len(predictions),
        predictions_sha256=digest(path),source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},
        input_sha256={str(p.relative_to(ROOT)):digest(p) for p in input_paths},reload_audits=audits,
        no_mask_or_clinical_label_inputs=True,patient_independence_proven=False,
        generator_selected_using_new_mask_results=False),indent=2)+'\n')
    print('All variants frozen before mask evaluation',flush=True)


if __name__ == '__main__':
    main()
