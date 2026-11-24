"""Audit actual fold checkpoints against independently reconstructed fit groups."""
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold,train_test_split
from experiment import ROOT,load_data,dump

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate-json',required=True);a=parser.parse_args()
    candidate=json.loads(__import__('pathlib').Path(a.candidate_json).read_text());feature=candidate['feature']
    frame,features,digest=load_data(feature);indices=np.flatnonzero(frame.split.eq('train').to_numpy());train=frame.iloc[indices]
    splitter=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=20261125)
    directory=ROOT/'runs'/feature/'candidate_crossvalidation'/('continuous_'+candidate['candidate_id']);results=[]
    for fold,(fit_local,outer_local) in enumerate(splitter.split(train,train.label,train.split_group)):
        fit_indices=indices[fit_local];outer_indices=indices[outer_local];fit=frame.iloc[fit_indices]
        patients=fit.groupby('split_group').label.agg(lambda x:x.mode().iloc[0])
        inner_ids,stop_ids=train_test_split(patients.index,test_size=.2,random_state=20261124+fold,stratify=patients)
        inner=fit[fit.split_group.isin(inner_ids)];stop=fit[fit.split_group.isin(stop_ids)]
        support=set(inner.patient_group);outside=set(stop.patient_group)|set(frame.iloc[outer_indices].patient_group)
        assert not support & outside
        expected_stats={}
        for name,values in features.items():
            if name=='region_mask':
                expected_stats[name]=(np.zeros(values.shape[-1],dtype='float32'),np.ones(values.shape[-1],dtype='float32'))
                continue
            fitting=values[inner.index.to_numpy()]
            if name=='regions' and 'region_mask' in features:
                valid=features['region_mask'][inner.index.to_numpy()]>.5
                if valid.any():fitting=fitting[valid]
            axes=tuple(range(fitting.ndim-1))
            expected_stats[name]=(fitting.mean(axes),np.maximum(fitting.std(axes),.05))
        for seed in range(3):
            ckpt=torch.load(directory/f'fold_{fold}'/f'seed_{seed}'/'model.pt',map_location='cpu',weights_only=False)
            assert ckpt['manifest_sha256']==digest
            for name,(expected_mean,expected_std) in expected_stats.items():
                assert np.allclose(ckpt['stats'][name]['mean'],expected_mean,atol=1e-6,rtol=1e-6),name
                assert np.allclose(ckpt['stats'][name]['std'],expected_std,atol=1e-6,rtol=1e-6),name
            if ckpt['cfg'].get('subtype_fusion',False) or ckpt['cfg'].get('taxonomy_only',False):
                disease_map=inner.groupby('disease').label.first()
                assert set(ckpt['diseases'])==set(inner.disease)
                assert ckpt['dims']['disease_to_class']==[int(disease_map[d]) for d in ckpt['diseases']]
                if ckpt['cfg'].get('taxonomy_residual',False):
                    subtype_support=inner.groupby('disease').patient_group.nunique()
                    assert ckpt['dims']['disease_patient_counts']==[int(subtype_support[d]) for d in ckpt['diseases']]
            if 'memory_audit' in ckpt:
                assert set(ckpt['memory_audit']['support_patient_groups'])==support
                assert ckpt['dims']['memory_count']==len(inner[['patient_group','label']].drop_duplicates())
                assert ckpt['memory_audit']['training_query_own_patient_excluded']
        saved=np.load(directory/f'fold_{fold}'/'outer_predictions.npz')
        assert np.array_equal(saved['indices'],outer_indices)
        results.append({'fold':fold,'inner_train_images':len(inner),'inner_stop_images':len(stop),'outer_images':len(outer_indices),'inner_patients':len(support),'outer_patients':frame.iloc[outer_indices].patient_group.nunique(),'seeds_checked':3})
    report={'passed':True,'candidate':candidate['candidate_id'],'manifest_sha256':digest,'checks':['All feature normalization exactly fits inner training images','No stop or outer patients in fit support','Subtype class map rebuilt from inner training labels','Subtype support counts use fit patients only when present','Patient memory support rebuilt inside each fit mask when present','Saved outer predictions cover the expected disjoint fold'],'folds':results}
    dump(directory/'provenance_audit.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
