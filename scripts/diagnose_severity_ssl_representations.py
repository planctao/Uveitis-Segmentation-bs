"""Compare completed domain encoders with matched original-DINO fit images."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,normalize

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def covariance_statistics(z):
    eigen=np.maximum(np.linalg.eigvalsh(np.cov(z,rowvar=False)),0)
    if eigen.sum()==0:return dict(effective_rank=0.,largest_variance_fraction=1.)
    w=eigen/eigen.sum();positive=w[w>0]
    return dict(effective_rank=float(np.exp(-np.sum(positive*np.log(positive)))),largest_variance_fraction=float(w.max()))


def describe(z):
    z=np.asarray(z,dtype=np.float64)
    angular=normalize(z)
    standardized=StandardScaler().fit_transform(angular)
    raw=covariance_statistics(z)
    return dict(raw_effective_rank=raw['effective_rank'],largest_raw_variance_fraction=raw['largest_variance_fraction'],
        angular_effective_rank=covariance_statistics(angular)['effective_rank'],
        standardized_angular_effective_rank=covariance_statistics(standardized)['effective_rank'],
        mean_coordinate_std=float(z.std(0).mean()),mean_feature_norm=float(np.linalg.norm(z,axis=1).mean()))


def main():
    root=ROOT/'outputs/severity_research_20261013'
    original_path=root/'embeddings/frozen_dino_gamma.npz'
    original=np.load(original_path,allow_pickle=False)
    original_features=original['embeddings'][0]
    original_uids=original['uids']
    records=[];matched_references={};inputs={str(original_path.relative_to(ROOT)):digest(original_path)}
    for family in ['swav','vicreg']:
        for complete_path in sorted((root/'domain_ssl'/family).glob('*/*/complete.json')):
            run=complete_path.parent
            complete=json.loads(complete_path.read_text())
            for name,expected in complete['source_sha256'].items():assert digest(ROOT/name)==expected,name
            plan=json.loads((run/'plan.json').read_text())
            rep_path=run/'final_representations.npz'
            rep=np.load(rep_path,allow_pickle=False)
            assert np.array_equal(original_uids,rep['uids'])
            train=np.isin(original_uids,plan['train_uids'])
            fold=complete['fold']
            train_hash=hashlib.sha256('\n'.join(original_uids[train]).encode()).hexdigest()
            if train_hash not in matched_references:
                matched_references[train_hash]=describe(original_features[train])
            reference=matched_references[train_hash]
            current=describe(rep['features'][0,train])
            projection=describe(rep['projections'][0,train])
            record=dict(family=family,seed=complete['seed'],fold=fold,n_fit=int(train.sum()),fit_uid_sha256=train_hash,
                std_ratio_to_original=current['mean_coordinate_std']/reference['mean_coordinate_std'],
                raw_rank_ratio_to_original=current['raw_effective_rank']/reference['raw_effective_rank'],
                angular_rank_ratio_to_original=current['angular_effective_rank']/reference['angular_effective_rank'])
            record.update({f'original_{k}':v for k,v in reference.items()})
            record.update({f'adapted_{k}':v for k,v in current.items()})
            record.update({f'projection_{k}':v for k,v in projection.items()})
            records.append(record)
            inputs[str(rep_path.relative_to(ROOT))]=digest(rep_path)
    out=root/'domain_ssl/representation_diagnostics';out.mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(out/'matched_encoder_diagnostics.csv',index=False)
    (out/'audit.json').write_text(json.dumps(dict(completed_states=len(records),planned_states_per_family=15,
        masks_read=False,early_stopping_or_checkpoint_selection=False,clinical_grading_validated=False,
        statistic_scope='Original fit images only; raw, L2 angular, and fit-standardized angular covariance',
        input_sha256=inputs,script_sha256=digest(Path(__file__)),records=records),indent=2)+'\n')
    print(pd.DataFrame(records)[['family','seed','fold','std_ratio_to_original','raw_rank_ratio_to_original',
        'original_raw_effective_rank','adapted_raw_effective_rank','projection_raw_effective_rank']].to_string(index=False),flush=True)


if __name__=='__main__':main()
