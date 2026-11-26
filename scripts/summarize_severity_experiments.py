"""Audit frozen experiment artifacts and generate evidence tables/figures."""
import hashlib
import json
from pathlib import Path
import sys

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    root=ROOT/'outputs/severity_research_20261013';out=root/'summary';out.mkdir(exist_ok=True)
    images=pd.read_csv(root/'data/image_groups.csv',dtype={'dhash':str}).set_index('uid')
    cache=np.load(root/'embeddings/frozen_dino_gamma.npz',allow_pickle=False)
    uid_index={uid:i for i,uid in enumerate(cache['uids'])}
    tables=[];robust=[];audits=[];responses=[];fold_tables=[]
    for family,seeds in [('baselines',[42]),('synthetic_calibration',[42,2026,3407]),('synthetic_controls',[42,2026,3407])]:
        for seed in seeds:
            run=root/family/f'seed{seed}'
            frozen=json.loads((run/'frozen_manifest.json').read_text())
            for name,digest in frozen['source_sha256'].items():
                assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
            prediction_path=run/'predictions_frozen_before_evaluation.csv'
            assert hashlib.sha256(prediction_path.read_bytes()).hexdigest()==frozen['predictions_sha256']
            prediction=pd.read_csv(prediction_path)
            assert not prediction.duplicated(['scope','fold','method','gamma','uid']).any()
            assert prediction.score.between(0,1).all()
            assert set(prediction.level)<={0,1,2,3}
            splits=json.loads((run/'splits.json').read_text());state_count=0;max_difference=0.
            for split in splits:
                train=images.loc[split['train_uids']];test=images.loc[split['test_uids']]
                assert not train.exclude_exact_group.any() and not test.exclude_exact_group.any()
                assert not set(train.pixel_sha256)&set(test.pixel_sha256)
                if split['scope']=='near_group_stress':assert not set(train.near_group)&set(test.near_group)
                test_prediction=prediction[(prediction.scope==split['scope'])&(prediction.fold==split['fold'])]
                for method,g in test_prediction.groupby('method'):
                    model=joblib.load(run/'states'/f'{split["scope"]}_{split["fold"]}_{method}.joblib')
                    state_count+=1
                    for gamma,gg in g.groupby('gamma'):
                        assert set(gg.uid)==set(test.index)
                        gg=gg.iloc[:3];indices=np.array([uid_index[u] for u in gg.uid])
                        gi=int(np.where(cache['gammas']==gamma)[0][0])
                        score,level=model.predict(cache['embeddings'][gi,indices],cache['image_features'][gi,indices])
                        difference=float(np.max(np.abs(score-gg.score.to_numpy())))
                        max_difference=max(max_difference,difference)
                        assert difference<1e-12 and np.array_equal(level,gg.level.to_numpy())
            for filename,collection in [('pooled_metrics.csv',tables),('gamma_robustness.csv',robust),('fold_metrics.csv',fold_tables)]:
                table=pd.read_csv(run/'evaluation'/filename);table['family']=family;table['seed']=seed;collection.append(table)
            if (run/'synthetic_dose_response.csv').exists():
                r=pd.read_csv(run/'synthetic_dose_response.csv');r['family']=family;r['seed']=seed;responses.append(r)
            audit=dict(family=family,seed=seed,fitted_states=state_count,prediction_rows=len(prediction),
                       source_hashes_match=True,frozen_prediction_hash_matches=True,
                       split_membership_checked=True,exact_duplicate_overlap=0,
                       reload_max_abs_score_difference=max_difference,
                       reload_levels_match=True,n_test_images=int(prediction.uid.nunique()))
            audits.append(audit);print(audit,flush=True)
    all_results=pd.concat(tables,ignore_index=True);all_results.to_csv(out/'all_pooled_metrics.csv',index=False)
    all_robust=pd.concat(robust,ignore_index=True);all_robust.to_csv(out/'all_gamma_robustness.csv',index=False)
    all_folds=pd.concat(fold_tables,ignore_index=True);all_folds.to_csv(out/'all_fold_metrics.csv',index=False)
    all_response=pd.concat(responses,ignore_index=True);all_response.to_csv(out/'all_synthetic_dose_response.csv',index=False)
    primary=all_results[all_results.scope=='main']
    seed_summary=primary.groupby(['family','method']).agg(n_seeds=('seed','nunique'),
        rho_mean=('union_rho','mean'),rho_sd=('union_rho','std'),rho_min=('union_rho','min'),rho_max=('union_rho','max'),
        fold_mean=('mean_fold_rho','mean'),brightness_rho_mean=('brightness_rho','mean')).reset_index()
    seed_summary.to_csv(out/'seed_summary.csv',index=False)
    # Explicit 2x2 comparison, without interpreting folds as independent patients.
    factors=['synthetic_direction','meanmatched_direction','synthetic_gamma_orthogonal_direction','gamma_orthogonal_direction']
    for seed in [42,2026,3407]:
        t=primary[primary.seed==seed].set_index('method')
        interaction=t.loc[factors[3],'union_rho']-t.loc[factors[1],'union_rho']-t.loc[factors[2],'union_rho']+t.loc[factors[0],'union_rho']
        print('seed',seed,'descriptive 2x2 interaction',float(interaction),flush=True)
    mechanism=seed_summary[seed_summary.family!='baselines'].sort_values('rho_mean')
    fig,axs=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    y=np.arange(len(mechanism));axs[0].barh(y,mechanism.rho_mean,color='#366e93')
    for _,row in primary[primary.family!='baselines'].iterrows():
        j=mechanism.index.get_loc(mechanism[mechanism.method==row.method].index[0])
        axs[0].plot(row.union_rho,j,'o',color='#d39341',markersize=4)
    axs[0].axvline(float(primary[primary.method=='mean'].union_rho.iloc[0]),color='#555',linestyle='--',label='Mean intensity')
    axs[0].set_yticks(y,mechanism.method);axs[0].set_xlabel('Held-out image mask-area Spearman rho')
    axs[0].set_title('Synthetic direction and projection controls');axs[0].legend()
    curves=all_folds[(all_folds.scope=='main')&(all_folds.seed==42)]
    for method in ['mean','synthetic_direction','synthetic_gamma_orthogonal_direction','gamma_orthogonal_direction']:
        g=curves[curves.method==method].sort_values('fold')
        axs[1].plot(g.fold,g.union_rho,'o-',label=method)
    axs[1].set_xlabel('Original image fold');axs[1].set_ylabel('Spearman rho')
    axs[1].set_title('Fold variation; patient separation unverified');axs[1].legend(fontsize=8)
    fig.savefig(out/'mechanism_comparison.png',dpi=180);fig.savefig(out/'mechanism_comparison.pdf');plt.close(fig)
    report=dict(experiments=audits,total_fitted_states=sum(a['fitted_states'] for a in audits),
                total_prediction_rows=sum(a['prediction_rows'] for a in audits),
                n_unique_evaluated=2055,clinical_severity_validated=False,patient_independence_verified=False,
                primary_baseline_direction_failure_preserved=True,main_mask_metric='union area fraction Spearman',
                seed_repetitions_are_not_independent_clinical_replicates=True,
                automated_goal_complete=False)
    (out/'final_artifact_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    # Snapshot the code/protocol needed to reproduce this stage.
    snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for path in [*ROOT.glob('scripts/*severity*.py'),*ROOT.glob('src/bs/severity*.py'),*ROOT.glob('docs/severity*20261013.md')]:
        target=snapshot/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())


if __name__=='__main__':main()
