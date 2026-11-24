"""Audit serialized repeat-CV plans, actual fitted checkpoints and outer predictions."""
import json
import numpy as np
import torch
from experiment import ROOT,load_data,metrics,dump


def main():
    torch.set_num_threads(4);frame,features,digest=load_data('multiscale_vit');base=ROOT/'runs/repeated_patient_cv'
    protocol=json.loads((base/'protocol.json').read_text());assert protocol['manifest_sha256']==digest
    original=set(np.flatnonzero(frame.split.eq('train').to_numpy()));checked=[]
    for candidate in protocol['candidates']:
        for repeat in protocol['new_outer_split_seeds']:
            seen=[]
            for fold in range(3):
                plan=json.loads((base/'split_plans'/f'repeat_{repeat}_fold_{fold}.json').read_text());assert plan['manifest_sha256']==digest
                fit,stop,outer=[np.asarray(plan[k],dtype=int) for k in ['fit_indices','stop_indices','outer_indices']]
                partitions=[set(fit),set(stop),set(outer)];assert set.union(*partitions)==original
                assert not partitions[0]&partitions[1] and not partitions[0]&partitions[2] and not partitions[1]&partitions[2]
                for key in ['patient_group','split_group']:
                    groups=[set(frame.iloc[ix][key]) for ix in [fit,stop,outer]]
                    assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
                tf=frame.iloc[fit];mapping=tf.groupby('disease').label.first();expected={}
                for name,values in features.items():
                    x=values[fit];axes=tuple(range(x.ndim-1));expected[name]=(x.mean(axes),np.maximum(x.std(axes),.05))
                directory=base/candidate['candidate_id']/f'repeat_{repeat}'/f'fold_{fold}'
                for seed in range(3):
                    c=torch.load(directory/f'seed_{seed}'/'model.pt',map_location='cpu',weights_only=False)
                    assert c['manifest_sha256']==digest and 'backbone_adapted' not in c
                    assert c['diseases']==sorted(tf.disease.unique())
                    assert c['dims']['disease_to_class']==[int(mapping[d]) for d in c['diseases']]
                    for name,(mean,std) in expected.items():
                        assert np.allclose(c['stats'][name]['mean'],mean,atol=1e-6,rtol=1e-6)
                        assert np.allclose(c['stats'][name]['std'],std,atol=1e-6,rtol=1e-6)
                saved=np.load(directory/'outer_predictions.npz');assert np.array_equal(saved['indices'],outer)
                p=saved['p'];assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=1e-5)
                actual=metrics(frame.iloc[outer].label,p);stored=json.loads((directory/'metrics.json').read_text())['metrics']
                assert abs(actual['accuracy']-stored['accuracy'])<1e-12 and abs(actual['macro_f1']-stored['macro_f1'])<1e-12
                seen.extend(outer.tolist());checked.append({'candidate':candidate['candidate_id'],'repeat':repeat,'fold':fold,'seed_checkpoints':3,'outer_images':len(outer)})
            assert len(seen)==len(original) and len(set(seen))==len(original) and set(seen)==original
    assert len(checked)==18
    dump(ROOT/'results/research_repeat_cv/verification.json',{'passed':True,'manifest_sha256':digest,'checks':['Every fold plan partitions only the original training set','Fit, stopping and outer patients and duplicate groups are disjoint','All 54 actual checkpoints fit normalization and subtype mapping on inner training only','Every training image has exactly one outer prediction per repeat and candidate','Saved probabilities reproduce every reported fold accuracy and macro-F1'],'checked_seed_fits':54,'folds':checked})
    print('Independent repeated-patient CV audit passed: 54 fits and 18 outer folds',flush=True)


if __name__=='__main__':main()
