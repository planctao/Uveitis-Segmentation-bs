"""Report all generator outcomes and paired deltas against the fixed baseline."""
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
from bs.severity_generator_sensitivity import GENERATOR_SPECS
from evaluate_severity_baselines import bootstrap_correlations


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = ROOT/'outputs/severity_research_20261013'
    run = root/'generator_sensitivity'
    manifest = json.loads((run/'frozen_manifest.json').read_text())
    path = run/'predictions_frozen_before_evaluation.csv'
    assert digest(path) == manifest['predictions_sha256']
    for name,expected in {**manifest['source_sha256'],**manifest['input_sha256']}.items():
        assert digest(ROOT/name) == expected,name
    pred = pd.read_csv(path)
    target_path = root/'data/mask_burden_evaluation_only.csv'
    targets = pd.read_csv(target_path).set_index('uid')
    original = pred[pred.gamma == 1.]
    metrics, folds, robust = [], [], []
    for (scope,method),g in original.groupby(['scope','method']):
        variants = list(GENERATOR_SPECS)
        pivot = g.pivot(index='uid',columns='variant',values='score').sort_index()[variants]
        truth = targets.loc[pivot.index]
        boot = bootstrap_correlations(pivot.to_numpy().T,truth.union_fraction.to_numpy())
        for k,variant in enumerate(variants):
            gv = g[g.variant == variant].set_index('uid').loc[pivot.index]
            base = g[g.variant == 'baseline'].set_index('uid').loc[pivot.index]
            ci = np.quantile(boot[k],[.025,.975])
            delta_ci = np.quantile(boot[k]-boot[0],[.025,.975])
            score_rho = rho(gv.score,truth.union_fraction)
            metrics.append(dict(scope=scope,method=method,variant=variant,n=len(gv),
                union_rho=score_rho,ci_low=float(ci[0]),ci_high=float(ci[1]),
                delta_vs_baseline=score_rho-rho(base.score,truth.union_fraction),
                delta_ci_low=float(delta_ci[0]),delta_ci_high=float(delta_ci[1]),
                rank_agreement_with_baseline=rho(gv.score,base.score),
                level_change_vs_baseline=float((gv.level != base.level).mean()),
                lesion1_rho=rho(gv.score,truth.lesion1_fraction),lesion2_rho=rho(gv.score,truth.lesion2_fraction),
                lesion2_presence_auc=float(roc_auc_score(truth.lesion2_fraction>0,gv.score))))
            for fold in sorted(gv.fold.unique()):
                f = gv[gv.fold == fold]
                ft = targets.loc[f.index]
                folds.append(dict(scope=scope,method=method,variant=variant,fold=fold,n=len(f),
                    union_rho=rho(f.score,ft.union_fraction),
                    lesion2_presence_auc=float(roc_auc_score(ft.lesion2_fraction>0,f.score))))
            for gamma in [.8,1.2]:
                changed = pred[(pred.scope==scope)&(pred.method==method)&(pred.variant==variant)&(pred.gamma==gamma)].set_index('uid').loc[pivot.index]
                robust.append(dict(scope=scope,method=method,variant=variant,gamma=gamma,
                    rank_consistency=rho(gv.score,changed.score),
                    level_change_fraction=float((gv.level != changed.level).mean())))
    out = run/'evaluation'
    out.mkdir(exist_ok=True)
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(out/'pooled_metrics.csv',index=False)
    pd.DataFrame(folds).to_csv(out/'fold_metrics.csv',index=False)
    pd.DataFrame(robust).to_csv(out/'gamma_robustness.csv',index=False)
    primary = metrics[(metrics.scope=='main')&(metrics.method=='gamma_orthogonal_direction')].set_index('variant').loc[list(GENERATOR_SPECS)]
    fig,axs = plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    y = np.arange(len(primary))
    axs[0].errorbar(primary.union_rho,y,xerr=[primary.union_rho-primary.ci_low,primary.ci_high-primary.union_rho],fmt='o',capsize=3)
    axs[0].set_yticks(y,primary.index)
    axs[0].set_xlabel('Held-out union area Spearman rho')
    axs[0].set_title('Fixed joint calibration; image bootstrap 95% CI')
    axs[1].plot(primary.lesion2_presence_auc,y,'o')
    axs[1].set_yticks(y,primary.index)
    axs[1].axvline(.5,color='gray',linestyle='--')
    axs[1].set_xlabel('Lesion 2 presence AUROC')
    axs[1].set_title('Auxiliary sign differs from union burden')
    for ax in axs:ax.invert_yaxis()
    fig.savefig(out/'generator_sensitivity.png',dpi=180)
    fig.savefig(out/'generator_sensitivity.pdf')
    plt.close(fig)
    (out/'evaluation_manifest.json').write_text(json.dumps(dict(
        predictions_sha256=manifest['predictions_sha256'],target_sha256=digest(target_path),
        evaluator_sha256=digest(Path(__file__)),n_bootstrap=1000,confidence_interval_unit='image',
        clinical_grading_validated=False,patient_independence_proven=False,
        original_generator_remains_primary=True),indent=2)+'\n')
    print(primary.to_string(),flush=True)


if __name__ == '__main__':
    main()
