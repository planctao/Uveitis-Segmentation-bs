"""Paired exploratory evaluation of all predeclared C4 calibration controls."""
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bs.severity_ordering import rho
from bs.severity_synthetic import SYNTHETIC_METHODS
from evaluate_severity_baselines import bootstrap_correlations

RUN = ROOT/'outputs/severity_research_20261013'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stability(part,baseline):
    return dict(rank_consistency=rho(part.score,baseline.score),
        level_change_fraction=float((part.level.to_numpy()!=baseline.level.to_numpy()).mean()),
        two_or_more_level_change_fraction=float((np.abs(part.level.to_numpy()-baseline.level.to_numpy())>=2).mean()),
        mean_absolute_score_change=float(np.abs(part.score.to_numpy()-baseline.score.to_numpy()).mean()))


def main():
    run = RUN/'rotation_pooling'
    inputs = {}
    def read_frozen(directory):
        path = directory/'frozen_manifest.json'
        m = json.loads(path.read_text())
        inputs[str(path.relative_to(ROOT))] = digest(path)
        for key in ['source_sha256','input_sha256']:
            for name,expected in m.get(key,{}).items():assert digest(ROOT/name)==expected,name
        path = directory/'predictions_frozen_before_evaluation.csv'
        assert digest(path)==m['predictions_sha256']
        inputs[str(path.relative_to(ROOT))] = digest(path)
        return m,pd.read_csv(path)
    frozen,pred = read_frozen(run)
    assert frozen['states']==80 and frozen['n_predictions']==len(pred)==328800
    assert frozen['no_mask_or_clinical_label_inputs'] and not frozen['primary_model_changed']
    for audit in frozen['reload_audits']:
        assert digest(ROOT/audit['state_path'])==audit['state_sha256']
        assert audit['all_test_conditions_reloaded'] and audit['c4_scores_and_levels_bitwise_equal']
    feature_manifest = json.loads((run/'feature_manifest.json').read_text())
    assert digest(run/'frozen_rotation_orbits.npz')==feature_manifest['features_sha256']
    assert max(feature_manifest['identity_max_errors'].values())==0
    assert not pred.duplicated(['branch','seed','uid','method','condition']).any()
    _,original_geometry = read_frozen(RUN/'geometric_stress')
    _,original_gamma = read_frozen(RUN/'synthetic_calibration/seed42')
    original_gamma = original_gamma[(original_gamma.scope=='main') & original_gamma.gamma.isin([.8,1.2])].copy()
    original_gamma['condition'] = original_gamma.gamma.map({.8:'gamma08',1.2:'gamma12'})
    original = pd.concat([original_geometry,original_gamma],ignore_index=True)
    original['branch'],original['seed'] = 'original',42
    all_predictions = pd.concat([original,pred],ignore_index=True)
    combinations = [('original',42),('rotation_score_mean',42)] + [('rotation_feature_mean',s) for s in [42,2026,3407]]
    conditions = frozen['conditions']
    assert len(conditions)==10
    frames,canonical = {},None
    for branch,seed in combinations:
        for method in SYNTHETIC_METHODS:
            for condition in conditions:
                part = all_predictions[(all_predictions.branch==branch)&(all_predictions.seed==seed)&
                    (all_predictions.method==method)&(all_predictions.condition==condition)].set_index('uid').sort_index()
                assert len(part)==part.index.nunique()==2055
                assert part.score.between(0,1).all() and part.level.isin([0,1,2,3]).all()
                if canonical is None:canonical,canonical_folds=part.index,part.fold
                assert part.index.equals(canonical) and part.fold.equals(canonical_folds)
                frames[branch,seed,method,condition] = part
    # All frozen inputs and complete condition sets pass before this label read.
    target_path = RUN/'data/mask_burden_evaluation_only.csv'
    targets = pd.read_csv(target_path).set_index('uid').loc[canonical]
    inputs[str(target_path.relative_to(ROOT))] = digest(target_path)
    keys = list(frames)
    scores = np.stack([frames[key].score.to_numpy() for key in keys])
    bootstrap = bootstrap_correlations(scores,targets.union_fraction.to_numpy())
    indices = {key:i for i,key in enumerate(keys)}
    pooled,folds = [],[]
    for i,(branch,seed,method,condition) in enumerate(keys):
        part = frames[branch,seed,method,condition]
        identity = frames[branch,seed,method,'identity']
        old_identity = frames['original',42,method,'identity']
        old_same = frames['original',42,method,condition]
        value = rho(part.score,targets.union_fraction)
        ci = np.quantile(bootstrap[i],[.025,.975])
        delta_old = bootstrap[i]-bootstrap[indices['original',42,method,'identity']]
        delta_same = bootstrap[i]-bootstrap[indices['original',42,method,condition]]
        delta_ci,delta_same_ci = np.quantile(delta_old,[.025,.975]),np.quantile(delta_same,[.025,.975])
        pooled.append(dict(branch=branch,seed=seed,method=method,condition=condition,n=2055,
            union_rho=value,ci_low=float(ci[0]),ci_high=float(ci[1]),
            delta_vs_original_identity=value-rho(old_identity.score,targets.union_fraction),
            delta_vs_original_identity_ci_low=float(delta_ci[0]),delta_vs_original_identity_ci_high=float(delta_ci[1]),
            delta_vs_original_same_condition=value-rho(old_same.score,targets.union_fraction),
            delta_vs_original_same_condition_ci_low=float(delta_same_ci[0]),delta_vs_original_same_condition_ci_high=float(delta_same_ci[1]),
            lesion1_rho=rho(part.score,targets.lesion1_fraction),lesion2_rho=rho(part.score,targets.lesion2_fraction),
            lesion2_presence_auc=float(roc_auc_score(targets.lesion2_fraction>0,part.score)),
            rank_consistency_with_original_identity=rho(part.score,old_identity.score),
            level_change_vs_original_identity=float((part.level.to_numpy()!=old_identity.level.to_numpy()).mean()),
            **stability(part,identity)))
        for fold in sorted(part.fold.unique()):
            selected = part.fold==fold
            g,t = part.loc[selected],targets.loc[selected]
            folds.append(dict(branch=branch,seed=seed,method=method,condition=condition,fold=fold,n=len(g),
                union_rho=rho(g.score,t.union_fraction),lesion1_rho=rho(g.score,t.lesion1_fraction),
                lesion2_rho=rho(g.score,t.lesion2_fraction),
                lesion2_presence_auc=float(roc_auc_score(t.lesion2_fraction>0,g.score)),
                delta_vs_original_identity=rho(g.score,t.union_fraction)-rho(old_identity.loc[selected,'score'],t.union_fraction),
                **stability(g,identity.loc[selected])))
    pooled,folds = pd.DataFrame(pooled),pd.DataFrame(folds)
    c4 = pooled[(pooled.branch!='original') & pooled.condition.isin(['rot90','rot180','rot270'])]
    assert (c4.level_change_fraction==0).all() and (c4.mean_absolute_score_change==0).all()
    assert np.allclose(c4.rank_consistency,1,atol=1e-12)
    old_path = RUN/'synthetic_calibration/seed42/evaluation/pooled_metrics.csv'
    old_metrics = pd.read_csv(old_path)
    inputs[str(old_path.relative_to(ROOT))] = digest(old_path)
    for method in SYNTHETIC_METHODS:
        current = pooled[(pooled.branch=='original')&(pooled.method==method)&(pooled.condition=='identity')].iloc[0]
        prior = old_metrics[(old_metrics.scope=='main')&(old_metrics.method==method)].iloc[0]
        assert abs(current.union_rho-prior.union_rho)<1e-12
        assert abs(current.lesion2_presence_auc-prior.lesion2_presence_auc)<1e-12
    response_path = run/'synthetic_dose_response.csv'
    assert digest(response_path)==frozen['synthetic_response_sha256']
    inputs[str(response_path.relative_to(ROOT))] = digest(response_path)
    responses = pd.read_csv(response_path)
    response_summary = []
    for key,g in responses.groupby(['branch','seed','method','split']):
        row = dict(zip(['branch','seed','method','split'],key));row['seed']=int(row['seed']);row['n']=int(g.n.sum())
        for column in ['strict_two_step_monotonic_fraction','high_gt_control_fraction',
                       'mean_localized_change','mean_control_change']:
            row[column] = float(np.average(g[column],weights=g.n))
        response_summary.append(row)
    out = run/'evaluation';out.mkdir(exist_ok=True)
    pooled.to_csv(out/'pooled_metrics.csv',index=False)
    folds.to_csv(out/'fold_metrics.csv',index=False)
    pd.DataFrame(response_summary).to_csv(out/'synthetic_response_summary.csv',index=False)
    fig,axes = plt.subplots(1,2,figsize=(13,4.7),constrained_layout=True)
    identity = pooled[pooled.condition=='identity']
    labels = ['Local direction','Mean matched','Joint gamma','Oriented PC1']
    colors = ['#444444','#cb7c27','#227f9d','#48a69b','#7e65a8']
    for j,((branch,seed),color) in enumerate(zip(combinations,colors)):
        rows = identity[(identity.branch==branch)&(identity.seed==seed)].set_index('method').loc[SYNTHETIC_METHODS]
        x = np.arange(4)+(j-2)*.12
        axes[0].errorbar(x,rows.union_rho,yerr=[rows.union_rho-rows.ci_low,rows.ci_high-rows.union_rho],
            fmt='o',capsize=2,color=color,label=f'{branch} {seed}')
    axes[0].set_xticks(np.arange(4),labels,rotation=25,ha='right')
    axes[0].set_ylabel('Image-level union-area Spearman')
    axes[0].legend(fontsize=7)
    axes[0].set_title('All calibrators; original images')
    shown_conditions = ['rot90','rot180','rot270','reflect_lr','gamma08','gamma12']
    for j,(branch,seed) in enumerate(combinations[:3]):
        g = pooled[(pooled.branch==branch)&(pooled.seed==seed)&(pooled.method=='gamma_orthogonal_direction')].set_index('condition').loc[shown_conditions]
        axes[1].plot(np.arange(6),g.level_change_fraction,marker='o',label=f'{branch} {seed}',color=colors[j])
    axes[1].set_xticks(np.arange(6),shown_conditions,rotation=35,ha='right')
    axes[1].set_ylabel('Quantile-level change fraction')
    axes[1].set_title('Joint gamma; C4 invariance is by construction')
    axes[1].legend(fontsize=7)
    fig.suptitle('Development controls; clinical severity unvalidated',fontsize=11)
    fig.savefig(out/'rotation_pooling_controls.png',dpi=190,bbox_inches='tight')
    fig.savefig(out/'rotation_pooling_controls.pdf',bbox_inches='tight');plt.close(fig)
    sources = [Path(__file__),ROOT/'scripts/evaluate_severity_baselines.py',ROOT/'src/bs/severity_ordering.py',
        ROOT/'docs/severity_rotation_pooling_protocol_20261013.md']
    outputs = ['pooled_metrics.csv','fold_metrics.csv','synthetic_response_summary.csv',
               'rotation_pooling_controls.png','rotation_pooling_controls.pdf']
    audit = dict(source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},input_sha256=inputs,
        output_sha256={name:digest(out/name) for name in outputs},n_images=2055,n_new_predictions=328800,
        n_states=80,n_pooled_rows=len(pooled),n_fold_rows=len(folds),
        all_conditions_frozen_before_target_read=True,original_identity_metrics_reproduced=True,
        c4_zero_change_is_structural_property=True,all_new_c4_score_and_level_changes_zero=True,
        n_bootstrap=1000,bootstrap_seed=20261013,confidence_interval_unit='image',
        multiple_candidate_adjustment=False,clinical_grading_validated=False,patient_independence_proven=False,
        original_primary_model_changed=False,clinical_lock_changed=False,
        posthoc_method_development=True,identity_results=identity.to_dict('records'),
        synthetic_response_summary=response_summary)
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(identity[['branch','seed','method','union_rho','ci_low','ci_high','delta_vs_original_identity',
        'delta_vs_original_identity_ci_low','delta_vs_original_identity_ci_high','lesion2_presence_auc',
        'level_change_vs_original_identity']].to_string(index=False),flush=True)
    print('Complete C4 evaluation:',len(pooled),'pooled rows;',len(folds),'fold rows',flush=True)


if __name__ == '__main__':
    main()
