"""Fit/predict only; no evaluation labels are loaded by this program."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bs.severity_ordering import METHODS,ImageOnlyOrderer


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT/'outputs/severity_research_20261013')
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    out=args.output/'baselines'/f'seed{args.seed}';out.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(args.output/'data/image_groups.csv',dtype={'dhash':str})
    cache=np.load(args.output/'embeddings/frozen_dino_gamma.npz',allow_pickle=False)
    assert np.array_equal(frame.uid.to_numpy(),cache['uids'])
    z,f=cache['embeddings'],cache['image_features']
    gammas=cache['gammas'];predictions=[];splits=[]
    for scope in ['main','near_group_stress']:
        for fold in sorted(frame.fold.unique()):
            eligible=~frame.exclude_exact_group.to_numpy()
            test=(frame.fold.to_numpy()==fold)&eligible
            train=(frame.fold.to_numpy()!=fold)&eligible
            if scope=='near_group_stress':
                train &= ~frame.near_group.isin(frame.loc[test,'near_group']).to_numpy()
            if not train.sum() or not test.sum():raise ValueError('Empty split')
            assert not set(frame.loc[train,'pixel_sha256'])&set(frame.loc[test,'pixel_sha256'])
            if scope=='near_group_stress':assert not set(frame.loc[train,'near_group'])&set(frame.loc[test,'near_group'])
            split=dict(scope=scope,fold=fold,n_train=int(train.sum()),n_test=int(test.sum()),
                       train_uids=frame.loc[train,'uid'].tolist(),test_uids=frame.loc[test,'uid'].tolist(),
                       exact_hash_overlap=0,near_group_overlap=int(len(set(frame.loc[train,'near_group'])&set(frame.loc[test,'near_group']))))
            splits.append(split)
            for method in METHODS:
                start=time.monotonic();model=ImageOnlyOrderer(method,args.seed).fit(z[0,train],f[0,train])
                for gi,gamma in enumerate(gammas):
                    scores,levels=model.predict(z[gi,test],f[gi,test])
                    for uid,score,level in zip(frame.loc[test,'uid'],scores,levels):
                        predictions.append(dict(uid=uid,scope=scope,fold=fold,method=method,gamma=float(gamma),score=float(score),level=int(level)))
                state_dir=out/'states';state_dir.mkdir(exist_ok=True)
                joblib.dump(model,state_dir/f'{scope}_{fold}_{method}.joblib',compress=3)
                (state_dir/f'{scope}_{fold}_{method}.json').write_text(json.dumps(model.diagnostics,indent=2)+'\n')
                print(f'{scope} {fold} {method} train={train.sum()} test={test.sum()} {time.monotonic()-start:.2f}s',flush=True)
    pd.DataFrame(predictions).to_csv(out/'predictions_frozen_before_evaluation.csv',index=False)
    (out/'splits.json').write_text(json.dumps(splits,indent=2)+'\n')
    sources=[Path(__file__),ROOT/'src/bs/severity_ordering.py',ROOT/'src/bs/severity_features.py',ROOT/'docs/severity_protocol_20261013.md']
    (out/'frozen_manifest.json').write_text(json.dumps(dict(seed=args.seed,
        no_mask_or_clinical_label_inputs=True,scopes=['main','near_group_stress'],
        predictions_sha256=hashlib.sha256((out/'predictions_frozen_before_evaluation.csv').read_bytes()).hexdigest(),
        source_sha256={str(s.relative_to(ROOT)):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources}),indent=2)+'\n')


if __name__=='__main__':main()
