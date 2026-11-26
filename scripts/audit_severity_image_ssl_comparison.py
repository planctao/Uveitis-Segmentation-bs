"""Paired, post hoc diagnostics of complete frozen image-SSL evaluations."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bs.severity_ordering import rho
from evaluate_severity_baselines import bootstrap_correlations

RUN = ROOT / 'outputs/severity_research_20261013'
METHODS = ['gamma_orthogonal_direction', 'meanmatched_direction',
           'synthetic_direction', 'synthetic_oriented_pc1']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def partial_rho(score, target, controls):
    x, y = rankdata(score), rankdata(target)
    design = np.column_stack([np.ones(len(x)), *[rankdata(v) for v in controls]])
    x -= design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y -= design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return float(np.corrcoef(x, y)[0, 1]) if min(x.std(), y.std()) > 1e-12 else np.nan


def main():
    inputs, frames, evaluations = {}, {}, {}
    candidates = [('dino', 42, RUN / 'synthetic_calibration/seed42')]
    candidates += [(family, seed, RUN / f'domain_ssl/{family}_evaluation/seed{seed}')
                   for family in ['swav', 'vicreg'] for seed in [42, 2026, 3407]
                   if (RUN / f'domain_ssl/{family}_evaluation/seed{seed}/evaluation/evaluation_manifest.json').exists()]
    target_path = RUN / 'data/mask_burden_evaluation_only.csv'
    target_hash = digest(target_path)
    canonical_uids, canonical_folds = None, None
    for family, seed, directory in candidates:
        frozen_path = directory / 'frozen_manifest.json'
        evaluation_path = directory / 'evaluation/evaluation_manifest.json'
        prediction_path = directory / 'predictions_frozen_before_evaluation.csv'
        frozen = json.loads(frozen_path.read_text())
        evaluated = json.loads(evaluation_path.read_text())
        assert digest(prediction_path) == frozen['predictions_sha256'] == evaluated['prediction_sha256']
        assert evaluated['target_sha256'] == target_hash
        assert digest(ROOT / 'scripts/evaluate_severity_baselines.py') == evaluated['evaluator_sha256']
        for name, expected in frozen['source_sha256'].items():
            assert digest(ROOT / name) == expected, name
        if family != 'dino':
            assert frozen['family'] == family and frozen['seed'] == seed
            assert frozen['five_fold_budget_verified'] and len(frozen['audits']) == 5
            assert {a['fold'] for a in frozen['audits']} == {'f1', 'f2', 'f3', 'f4', 'f5'}
        pred = pd.read_csv(prediction_path)
        assert not pred.duplicated(['scope', 'fold', 'uid', 'method', 'gamma']).any()
        pooled_path = directory / 'evaluation/pooled_metrics.csv'
        fold_path = directory / 'evaluation/fold_metrics.csv'
        evaluations[family, seed] = (pd.read_csv(pooled_path), pd.read_csv(fold_path))
        for path in [frozen_path, evaluation_path, prediction_path, pooled_path, fold_path]:
            inputs[str(path.relative_to(ROOT))] = digest(path)
        prefix = '' if family == 'dino' else f'{family}_encoder_'
        for method in METHODS:
            part = pred[(pred.scope == 'main') & (pred.gamma == 1.) & (pred.method == prefix + method)].set_index('uid').sort_index()
            assert len(part) == part.index.nunique() == 2055
            assert np.isfinite(part.score).all() and part.score.between(0, 1).all()
            assert part.level.isin([0, 1, 2, 3]).all()
            if canonical_uids is None:
                canonical_uids, canonical_folds = part.index, part.fold
            assert part.index.equals(canonical_uids) and part.fold.equals(canonical_folds)
            frames[family, seed, method] = part
    # Open targets only after every frozen prediction and complete seed passes.
    targets = pd.read_csv(target_path).set_index('uid').loc[canonical_uids]
    cache_path = RUN / 'embeddings/frozen_dino_gamma.npz'
    cache = np.load(cache_path, allow_pickle=False)
    brightness = pd.Series(cache['image_features'][0, :, 0], index=cache['uids']).loc[canonical_uids].to_numpy()
    inputs[str(target_path.relative_to(ROOT))] = target_hash
    inputs[str(cache_path.relative_to(ROOT))] = digest(cache_path)
    keys = list(frames)
    scores = np.stack([frames[k].score.to_numpy() for k in keys])
    boot = bootstrap_correlations(scores, targets.union_fraction.to_numpy())
    index = {key: i for i, key in enumerate(keys)}
    positive = targets.lesion2_fraction.to_numpy() > 0
    records, comparisons, fold_comparisons = [], [], []
    for i, (family, seed, method) in enumerate(keys):
        part, score = frames[family, seed, method], scores[i]
        union_rho = rho(score, targets.union_fraction)
        auc = float(roc_auc_score(positive, score))
        lo, hi = np.quantile(boot[i], [.025, .975])
        pooled, folds = evaluations[family, seed]
        full_method = method if family == 'dino' else f'{family}_encoder_{method}'
        reference = pooled[(pooled.scope == 'main') & (pooled.method == full_method)].iloc[0]
        assert abs(union_rho - reference.union_rho) < 1e-12
        assert abs(auc - reference.lesion2_presence_auc) < 1e-12
        for fold in sorted(part.fold.unique()):
            selected = part.fold.to_numpy() == fold
            saved = folds[(folds.scope == 'main') & (folds.fold == fold) & (folds.method == full_method)].iloc[0]
            assert abs(rho(score[selected], targets.union_fraction.to_numpy()[selected]) - saved.union_rho) < 1e-12
            assert abs(float(roc_auc_score(positive[selected], score[selected])) - saved.lesion2_presence_auc) < 1e-12
        records.append(dict(family=family, seed=seed, method=method, n=len(part), union_rho=union_rho,
            ci_low=float(lo), ci_high=float(hi), lesion2_presence_auc=auc, n_lesion2_positive=int(positive.sum()),
            positive_only_lesion2_rho=rho(score[positive], targets.lesion2_fraction.to_numpy()[positive]),
            partial_lesion2_given_lesion1=partial_rho(score, targets.lesion2_fraction, [targets.lesion1_fraction]),
            partial_lesion2_given_lesion1_and_brightness=partial_rho(score, targets.lesion2_fraction, [targets.lesion1_fraction, brightness]),
            brightness_rho=rho(score, brightness), n_unique_scores=int(part.score.nunique())))
        references = [('dino', 42, method)] if family != 'dino' else []
        if family == 'vicreg' and ('swav', seed, method) in index:
            references.append(('swav', seed, method))
        for ref in references:
            j = index[ref]
            delta = union_rho - rho(scores[j], targets.union_fraction)
            low, high = np.quantile(boot[i] - boot[j], [.025, .975])
            differences = []
            for fold in sorted(part.fold.unique()):
                selected = part.fold.to_numpy() == fold
                left = rho(score[selected], targets.union_fraction.to_numpy()[selected])
                right = rho(scores[j, selected], targets.union_fraction.to_numpy()[selected])
                differences.append(left - right)
                fold_comparisons.append(dict(family=family, seed=seed, method=method, reference_family=ref[0],
                    reference_seed=ref[1], fold=fold, n=int(selected.sum()), union_rho=left,
                    reference_union_rho=right, delta_union_rho=left-right))
            comparisons.append(dict(family=family, seed=seed, method=method, reference_family=ref[0],
                reference_seed=ref[1], delta_union_rho=delta, delta_ci_low=float(low), delta_ci_high=float(high),
                n_folds_positive_delta=int(np.sum(np.asarray(differences) > 0)), n_folds=5,
                score_rank_consistency=rho(score, scores[j]),
                level_change_fraction=float((part.level.to_numpy() != frames[ref].level.to_numpy()).mean()),
                delta_lesion2_presence_auc=auc-float(roc_auc_score(positive, scores[j]))))
    out = RUN / 'domain_ssl/image_ssl_comparison'
    out.mkdir(exist_ok=True)
    for name, values in [('sign_diagnostics.csv', records), ('paired_differences.csv', comparisons),
                         ('paired_fold_differences.csv', fold_comparisons)]:
        pd.DataFrame(values).to_csv(out / name, index=False)
    sources = [Path(__file__), ROOT / 'docs/severity_image_ssl_comparison_plan_20261013.md',
               ROOT / 'scripts/evaluate_severity_baselines.py', ROOT / 'src/bs/severity_ordering.py']
    audit = dict(posthoc_analysis=True, model_refitted_or_selected=False, partial_fold_proxy_evaluation=False,
        clinical_grading_validated=False, patient_independence_proven=False, n_bootstrap=1000,
        bootstrap_seed=20261013, confidence_interval_unit='image', multiple_candidate_adjustment=False,
        input_sha256=inputs, source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
        completed_seeds=[dict(family=f, seed=s) for f, s, _ in candidates],
        masks_opened_after_prediction_audits=True, original_pooled_and_fold_metrics_reproduced=True,
        sign_diagnostics=records, paired_differences=comparisons)
    audit['output_sha256'] = {name: digest(out / name) for name in
        ['sign_diagnostics.csv', 'paired_differences.csv', 'paired_fold_differences.csv']}
    (out / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(pd.DataFrame(records).to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__ == '__main__':
    main()
