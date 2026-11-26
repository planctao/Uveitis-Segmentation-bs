"""Evaluate already frozen registration stress against the original fits."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bs.severity_ordering import rho
from evaluate_severity_baselines import bootstrap_correlations


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = ROOT / 'outputs/severity_research_20261013'
    all_metrics, all_folds, all_robustness, input_hashes = [], [], [], {}
    for seed in [42, 2026, 3407]:
        run = root / 'registration_stress' / f'seed{seed}'
        manifest = json.loads((run / 'frozen_manifest.json').read_text())
        pred_path = run / 'predictions_frozen_before_evaluation.csv'
        assert digest(pred_path) == manifest['predictions_sha256']
        for name, expected in {**manifest['source_sha256'], **manifest['input_sha256']}.items():
            assert digest(ROOT / name) == expected, name
        input_hashes[str(pred_path.relative_to(ROOT))] = digest(pred_path)
        pred = pd.read_csv(pred_path)
        main_path = root / 'synthetic_calibration' / f'seed{seed}' / 'predictions_frozen_before_evaluation.csv'
        main_manifest = json.loads((main_path.parent / 'frozen_manifest.json').read_text())
        assert digest(main_path) == main_manifest['predictions_sha256']
        main_pred = pd.read_csv(main_path)
        main_pred = main_pred[(main_pred.scope == 'main') & (main_pred.gamma == 1.)]
        # Targets enter only after both prediction hashes and input provenance pass.
        targets = pd.read_csv(root / 'data/mask_burden_evaluation_only.csv').set_index('uid')
        original = pred[pred.gamma == 1.]
        pivot = original.pivot(index='uid', columns='method', values='score').sort_index()
        reference = main_pred.pivot(index='uid', columns='method', values='score').loc[pivot.index, pivot.columns]
        y = targets.loc[pivot.index, 'union_fraction'].to_numpy()
        n_methods = len(pivot.columns)
        boots = bootstrap_correlations(np.concatenate([pivot.to_numpy().T, reference.to_numpy().T]), y)
        for k, method in enumerate(pivot.columns):
            saved = original[original.method == method].set_index('uid').loc[pivot.index]
            main_saved = main_pred[main_pred.method == method].set_index('uid').loc[pivot.index]
            assert np.array_equal(saved.fold, main_saved.fold)
            ci = np.quantile(boots[k], [.025, .975])
            delta_ci = np.quantile(boots[k] - boots[k + n_methods], [.025, .975])
            truth = targets.loc[pivot.index]
            stress_rho, main_rho = rho(saved.score, y), rho(main_saved.score, y)
            all_metrics.append(dict(seed=seed, method=method, n=len(saved),
                stress_union_rho=stress_rho, main_union_rho=main_rho,
                delta_rho=stress_rho-main_rho, ci_low=float(ci[0]), ci_high=float(ci[1]),
                delta_ci_low=float(delta_ci[0]), delta_ci_high=float(delta_ci[1]),
                main_stress_rank_agreement=rho(saved.score, main_saved.score),
                main_stress_level_change=float((saved.level != main_saved.level).mean()),
                lesion2_presence_auc=float(roc_auc_score(truth.lesion2_fraction > 0, saved.score))))
            for fold in sorted(saved.fold.unique()):
                group = saved[saved.fold == fold]
                ref = main_saved.loc[group.index]
                target = targets.loc[group.index, 'union_fraction']
                all_folds.append(dict(seed=seed, fold=fold, method=method, n=len(group),
                    stress_union_rho=rho(group.score, target), main_union_rho=rho(ref.score, target),
                    delta_rho=rho(group.score, target)-rho(ref.score, target)))
            for gamma in [.8, 1.2]:
                changed = pred[(pred.method == method) & (pred.gamma == gamma)].set_index('uid').loc[pivot.index]
                all_robustness.append(dict(seed=seed, method=method, gamma=gamma,
                    rank_consistency=rho(saved.score, changed.score),
                    level_change_fraction=float((saved.level != changed.level).mean())))
    out = root / 'registration_stress' / 'summary'
    out.mkdir(exist_ok=True)
    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(out / 'pooled_metrics.csv', index=False)
    pd.DataFrame(all_folds).to_csv(out / 'fold_metrics.csv', index=False)
    pd.DataFrame(all_robustness).to_csv(out / 'gamma_robustness.csv', index=False)
    splits = json.loads((root / 'registration_stress/seed42/splits.json').read_text())
    groups = pd.read_csv(root / 'registration_audit/image_registration_groups.csv').set_index('uid')
    target = pd.read_csv(root / 'data/mask_burden_evaluation_only.csv').set_index('uid')
    pred = pd.read_csv(root / 'registration_stress/seed42/predictions_frozen_before_evaluation.csv')
    score = pred[(pred.method == 'gamma_orthogonal_direction') & (pred.gamma == 1.)].set_index('uid').score.sort_index()
    # Group resampling reflects discovered links only, not unobserved patients.
    group_rows = groups.loc[score.index].reset_index().groupby('registration_group').uid.apply(list).tolist()
    rng = np.random.default_rng(20261013)
    grouped_boot = []
    for _ in range(1000):
        sample = [uid for k in rng.integers(0, len(group_rows), len(group_rows)) for uid in group_rows[k]]
        grouped_boot.append(rho(score.loc[sample], target.loc[sample, 'union_fraction']))
    report = dict(n_scoring_states=60, n_predictions=3 * 2055 * 4 * 3,
        fold_exclusions=[{k: v for k, v in s.items() if not k.endswith('_uids')} for s in splits],
        detected_registration_group_bootstrap=dict(method='gamma_orthogonal_direction', seed=42,
            n_groups=len(group_rows), n_bootstrap=1000, ci=np.quantile(grouped_boot, [.025, .975]).tolist(),
            true_patient_groups=False),
        prediction_inputs_sha256=input_hashes,
        target_sha256=digest(root / 'data/mask_burden_evaluation_only.csv'),
        evaluator_sha256=digest(Path(__file__)),
        clinical_grading_validated=False, patient_independence_proven=False,
        dec_and_swav_refit_in_this_stress=False,
        confidence_interval_unit='Image for all methods; discovered registration groups for seed42 joint candidate only')
    (out / 'evaluation_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(metrics.to_string(index=False), flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
