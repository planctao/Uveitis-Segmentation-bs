"""Replay all scores of completed SwAV states without opening mask targets."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def effective_rank(z):
    x = np.asarray(z,dtype=np.float64)
    eigen = np.maximum(np.linalg.eigvalsh(np.cov(x,rowvar=False)),0)
    weights = eigen/eigen.sum()
    weights = weights[weights>0]
    return float(np.exp(-np.sum(weights*np.log(weights))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = ROOT/'outputs/severity_research_20261013/domain_ssl'
    paths = sorted((root/'swav').glob('*/*/complete.json'))
    if args.require_complete:
        assert len(paths)==15, f'Only {len(paths)}/15 complete; do not infer missing folds'
    records = []
    total_predictions = 0
    for complete_path in paths:
        run = complete_path.parent
        complete = json.loads(complete_path.read_text())
        for name,expected in complete['source_sha256'].items():
            assert digest(ROOT/name)==expected,name
        path = run/'predictions_frozen_before_evaluation.csv'
        assert digest(path)==complete['predictions_sha256']
        pred = pd.read_csv(path)
        plan = json.loads((run/'plan.json').read_text())
        rep = np.load(run/'final_representations.npz',allow_pickle=False)
        saved = np.load(run/'prototype_order.npz',allow_pickle=False)
        train = np.isin(rep['uids'],plan['train_uids'])
        test = np.isin(rep['uids'],plan['test_uids'])
        checkpoint = torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
        assert np.array_equal(checkpoint['model']['prototypes.weight'].numpy(),rep['prototype_weights'])
        trace = pd.read_csv(run/'training_trace.csv')
        assert len(trace)==20 and complete['epochs']==20
        assert int((trace.successful_updates+trace.amp_skips).sum())==20*(int(train.sum())//8)
        assert trace.encoder_stem_max_abs_update.iloc[-1]>0
        max_error = 0.
        for method in ['synthetic_direction','meanmatched_direction','gamma_orthogonal_direction','synthetic_oriented_pc1']:
            calibrator = joblib.load(run/f'swav_encoder_{method}.joblib')
            assert calibrator.diagnostics['n_fit']==int(train.sum())
            for gi,gamma in enumerate([1.,.8,1.2]):
                scores,levels = calibrator.predict(rep['features'][gi,test])
                frozen = pred[(pred.method==f'swav_encoder_{method}')&(pred.gamma==gamma)].set_index('uid').loc[rep['uids'][test]]
                error = float(np.max(np.abs(scores-frozen.score)))
                max_error = max(max_error,error)
                assert error<1e-12 and np.array_equal(levels,frozen.level)
        # Reconstruct prototype ordering and fit-fold CDF, then all test outputs.
        probabilities = []
        for projected in rep['projections'][:3]:
            logits = projected@rep['prototype_weights'].T/.1
            logits -= logits.max(1,keepdims=True)
            q = np.exp(logits)
            probabilities.append(q/q.sum(1,keepdims=True))
        fit_q = probabilities[0][train]
        anchor = joblib.load(run/'swav_encoder_synthetic_direction.joblib')
        anchor_means = (fit_q.T@anchor.raw_score(rep['features'][0,train]))/np.maximum(fit_q.sum(0),1e-12)
        order = np.argsort(np.argsort(anchor_means))
        assert np.array_equal(order,saved['order'])
        assert np.allclose(anchor_means,saved['anchor_means'],atol=1e-10,rtol=1e-10)
        raw_fit = fit_q@order
        sorted_scores = np.sort(raw_fit)
        cutpoints = np.quantile(raw_fit,[.25,.5,.75])
        assert np.allclose(sorted_scores,saved['sorted_scores'],atol=1e-10,rtol=0)
        assert np.allclose(cutpoints,saved['cutpoints'],atol=1e-10,rtol=0)
        for gi,gamma in enumerate([1.,.8,1.2]):
            raw = probabilities[gi][test]@order
            scores = (np.searchsorted(sorted_scores,raw,'left')+np.searchsorted(sorted_scores,raw,'right'))/(2*len(sorted_scores))
            levels = np.searchsorted(cutpoints,raw,'right')
            frozen = pred[(pred.method=='swav_ordered_prototypes')&(pred.gamma==gamma)].set_index('uid').loc[rep['uids'][test]]
            error = float(np.max(np.abs(scores-frozen.score)))
            max_error = max(max_error,error)
            assert error<1e-12 and np.array_equal(levels,frozen.level)
        total_predictions += len(pred)
        counts = np.bincount(fit_q.argmax(1),minlength=64)
        mass = fit_q.sum(0)
        p = mass/mass.sum()
        records.append(dict(seed=complete['seed'],fold=complete['fold'],n_fit=int(train.sum()),n_test=int(test.sum()),
            n_predictions=len(pred),max_rank_error=max_error,prototype_order_and_cutpoints_rederived=True,
            prototype_weights_match_checkpoint=True,all_calibration_states_reloaded=True,
            successful_updates=complete['optimizer_updates'],amp_skips=complete['amp_skips'],
            hard_nonempty_prototypes=int((counts>0).sum()),max_hard_prototype_fraction=float(counts.max()/counts.sum()),
            soft_mass_effective_prototype_count=float(np.exp(-np.sum(p*np.log(p)))),
            encoder_effective_covariance_rank=effective_rank(rep['features'][0,train]),
            projection_effective_covariance_rank=effective_rank(rep['projections'][0,train]),
            last_epoch_loss=float(trace.mean_loss.iloc[-1]),prediction_sha256=digest(path)))
        print(run.relative_to(root),'all scores replayed',flush=True)
    out = root/'swav_state_audit'
    out.mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(out/'completed_fold_diagnostics.csv',index=False)
    (out/'audit.json').write_text(json.dumps(dict(completed_folds=len(records),planned_folds=15,
        all_planned_folds_complete=len(records)==15,n_predictions_replayed=total_predictions,
        masks_read=False,clinical_grading_validated=False,patient_independence_proven=False,
        image_encoder_reload_scope='Separate five-fold collector verifies image-to-encoder outputs; this audit verifies all cached-to-score outputs',
        source_sha256=digest(Path(__file__)),audits=records),indent=2)+'\n')


if __name__=='__main__':main()
