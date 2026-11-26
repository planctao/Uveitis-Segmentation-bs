"""Replay every DEC/AE ordered-cluster prediction from saved centroids."""
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bs.severity_ssl import student_assignments


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(4)
    root=ROOT/'outputs/severity_research_20261013/domain_ssl'
    results=[]
    for seed in [42,2026,3407]:
        for fold in ['f1','f2','f3','f4','f5']:
            run=root/'dec'/f'seed{seed}'/fold
            complete=json.loads((run/'complete.json').read_text())
            for name,expected in complete['source_sha256'].items():assert digest(ROOT/name)==expected,name
            path=run/'predictions_frozen_before_evaluation.csv'
            assert digest(path)==complete['predictions_sha256']
            pred=pd.read_csv(path)
            plan=json.loads((run/'plan.json').read_text())
            for branch in ['ae','dec']:
                rep=np.load(run/f'{branch}_representations.npz',allow_pickle=False)
                saved=np.load(run/f'{branch}_cluster_order.npz',allow_pickle=False)
                checkpoint=torch.load(run/f'{branch}_final.pt',map_location='cpu',weights_only=False)
                assert np.array_equal(saved['centers'],checkpoint['centers'])
                calibrator=joblib.load(run/f'{branch}_synthetic_direction.joblib')
                order=np.argsort(np.argsort(calibrator.raw_score(saved['centers'])))
                assert np.array_equal(order,saved['order'])
                train=np.isin(rep['uids'],plan['train_uids'])
                test=np.isin(rep['uids'],plan['test_uids'])
                qtrain=student_assignments(torch.from_numpy(rep['features'][0,train]),torch.from_numpy(saved['centers'])).numpy()
                training_scores=np.sort(qtrain@order)
                assert np.array_equal(training_scores,saved['sorted_scores'])
                max_error=0.
                for gi,gamma in enumerate([1.,.8,1.2]):
                    q=student_assignments(torch.from_numpy(rep['features'][gi,test]),torch.from_numpy(saved['centers'])).numpy()
                    raw=q@order
                    ranks=(np.searchsorted(training_scores,raw,'left')+np.searchsorted(training_scores,raw,'right'))/(2*len(training_scores))
                    levels=order[q.argmax(1)]
                    frozen=pred[(pred.method==f'{branch}_ordered_clusters')&(pred.gamma==gamma)].set_index('uid').loc[rep['uids'][test]]
                    error=float(np.max(np.abs(ranks-frozen.score)))
                    max_error=max(max_error,error)
                    assert error<1e-12 and np.array_equal(levels,frozen.level)
                results.append(dict(seed=seed,fold=fold,branch=branch,n_predictions=int(test.sum())*3,
                    max_rank_error=max_error,centroids_match_neural_checkpoint=True,
                    synthetic_cluster_order_rederived=True,train_rank_reference_rederived=True,
                    state_sha256=digest(run/f'{branch}_cluster_order.npz')))
    out=root/'dec_summary/ordered_cluster_reload_audit.json'
    out.write_text(json.dumps(dict(n_states=len(results),n_predictions=sum(r['n_predictions'] for r in results),
        masks_read=False,clinical_grading_validated=False,levels_semantics='Rank of argmax Student-t cluster; not quantile severity grades',
        source_sha256=digest(Path(__file__)),audits=results),indent=2)+'\n')
    print('All ordered-cluster states replayed',len(results),sum(r['n_predictions'] for r in results),flush=True)


if __name__=='__main__':main()
