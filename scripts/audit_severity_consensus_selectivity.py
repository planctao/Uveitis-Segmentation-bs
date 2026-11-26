"""Post-hoc matched-subset controls for the fixed generator disagreement."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bs.severity_generator_consensus import CONSENSUS_METHODS
from evaluate_severity_generator_consensus import statistics
from evaluate_severity_baselines import bootstrap_correlations


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = ROOT / 'outputs/severity_research_20261013'
    run = root / 'generator_consensus/seed42'
    frozen = json.loads((run / 'frozen_manifest.json').read_text())
    evaluation_path = run / 'evaluation/evaluation_manifest.json'
    evaluation = json.loads(evaluation_path.read_text())
    assert digest(ROOT / 'scripts/evaluate_severity_generator_consensus.py') == evaluation['evaluator_sha256']
    pred_path = run / 'predictions_frozen_before_evaluation.csv'
    detail_path = run / 'generator_disagreement_frozen_before_evaluation.csv'
    assert digest(pred_path) == frozen['predictions_sha256'] == evaluation['prediction_sha256']
    assert digest(detail_path) == frozen['disagreement_sha256'] == evaluation['disagreement_sha256']
    pred, detail = pd.read_csv(pred_path), pd.read_csv(detail_path)
    pred, detail = pred[pred.gamma == 1.], detail[detail.gamma == 1.]
    target_path = root / 'data/mask_burden_evaluation_only.csv'
    assert digest(target_path) == evaluation['target_sha256']
    targets = pd.read_csv(target_path).set_index('uid')
    cache = np.load(root / 'embeddings/frozen_dino_gamma.npz', allow_pickle=False)
    brightness = pd.Series(cache['image_features'][0, :, 0], index=cache['uids'])
    comparisons, pair_controls = [], []
    for scope in ['main', 'registration_group_stress']:
        d = detail[detail.scope == scope].set_index('uid').sort_index()
        scores = pred[pred.scope == scope].pivot(index='uid', columns='method', values='score').loc[d.index]
        scores.insert(0, 'original_generator', d.baseline_score)
        methods = ['original_generator', *CONSENSUS_METHODS]
        bins = pd.cut(d.rank_range, [0, .1, .25, .5, np.inf], right=False,
                      labels=['[0,.10)', '[.10,.25)', '[.25,.50)', '[.50,1]'])
        subsets = [('all', np.ones(len(d), dtype=bool)), ('same_level', d.all_generators_agree_level),
                   ('different_levels', ~d.all_generators_agree_level)]
        subsets.extend((label, bins == label) for label in ['[0,.10)', '[.10,.25)', '[.25,.50)', '[.50,1]'])
        for label, selected in subsets:
            idx = d.index[selected]
            if len(idx) < 3:
                continue
            t = targets.loc[idx]
            boot = bootstrap_correlations(scores.loc[idx, methods].to_numpy().T, t.union_fraction.to_numpy()) if len(idx) >= 20 else None
            for j, method in enumerate(methods):
                row = dict(scope=scope, group=label, method=method, n=len(idx),
                    **statistics(scores.loc[idx, method].to_numpy(), t, brightness.loc[idx].to_numpy()))
                if boot is not None:
                    ci, delta = np.quantile(boot[j], [.025, .975]), np.quantile(boot[j] - boot[0], [.025, .975])
                    row.update(ci_low=float(ci[0]), ci_high=float(ci[1]),
                               delta_ci_low=float(delta[0]), delta_ci_high=float(delta[1]))
                comparisons.append(row)
        variant_columns = [f'{v}_score' for v in frozen['generator_variants']]
        for fold in ['f1', 'f2', 'f3', 'f4', 'f5']:
            f = d[d.fold == fold]
            area = targets.loc[f.index, 'union_fraction'].to_numpy()
            ia, ib = np.triu_indices(len(f), 1)
            truth_delta = area[ia] - area[ib]
            eligible = np.abs(truth_delta) >= .01
            ia, ib, truth_delta = ia[eligible], ib[eligible], truth_delta[eligible]
            matrix = f[variant_columns].to_numpy()
            differences = matrix[ia] - matrix[ib]
            positive, negative = (differences > 0).all(1), (differences < 0).all(1)
            unanimous = positive | negative
            baseline_correct = differences[:, 0] * truth_delta > 0
            common_sign = positive.astype(np.int8) - negative.astype(np.int8)
            assert np.array_equal(baseline_correct[unanimous], (common_sign * truth_delta > 0)[unanimous])
            pair_controls.append(dict(scope=scope, fold=fold, n_dependent_pairs=len(ia),
                n_baseline_correct_all_pairs=int(baseline_correct.sum()), n_unanimous_pairs=int(unanimous.sum()),
                n_baseline_correct_unanimous_pairs=int(baseline_correct[unanimous].sum()),
                unanimous_correctness_identical_to_original_generator=True))
    out = run / 'selectivity_audit'
    out.mkdir(exist_ok=True)
    table = pd.DataFrame(comparisons)
    table.to_csv(out / 'matched_subset_original_controls.csv', index=False)
    counts = pd.DataFrame(pair_controls)
    counts.to_csv(out / 'original_generator_pair_controls.csv', index=False)
    pooled = counts.groupby('scope')[['n_dependent_pairs', 'n_baseline_correct_all_pairs', 'n_unanimous_pairs',
                                     'n_baseline_correct_unanimous_pairs']].sum().reset_index()
    pooled['baseline_all_pair_area_direction_fraction'] = pooled.n_baseline_correct_all_pairs / pooled.n_dependent_pairs
    pooled['baseline_unanimous_pair_area_direction_fraction'] = pooled.n_baseline_correct_unanimous_pairs / pooled.n_unanimous_pairs
    pooled.to_csv(out / 'pooled_pair_controls.csv', index=False)
    for row in evaluation['dependent_pair_summary']:
        control = pooled[pooled.scope == row['scope']].iloc[0]
        assert control.n_dependent_pairs == row['n_dependent_pairs']
        assert control.n_unanimous_pairs == row['n_unanimous_pairs']
        assert control.n_baseline_correct_unanimous_pairs == row['n_correct_unanimous_pairs']
    report = dict(posthoc_descriptive_audit=True, models_refitted=False, model_or_threshold_selected=False,
        clinical_grading_validated=False, patient_independence_proven=False, n_bootstrap=1000,
        confidence_interval_unit='image', pair_combinations_are_dependent=True,
        source_sha256={str(Path(__file__).relative_to(ROOT)): digest(Path(__file__)),
                       'scripts/evaluate_severity_generator_consensus.py': evaluation['evaluator_sha256']},
        input_sha256={str(path.relative_to(ROOT)): digest(path) for path in [pred_path, detail_path, target_path, evaluation_path]},
        interpretation='Unanimous pair correctness equals the original generator on that same subset by construction; improvement over all pairs is selection, not aggregation.',
        pair_controls=pooled.to_dict('records'))
    (out / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(table[(table.scope == 'main') & table.group.isin(['all', 'same_level', 'different_levels'])]
          [['group', 'method', 'n', 'union_rho', 'delta_ci_low', 'delta_ci_high']].to_string(index=False))
    print(pooled.to_string(index=False))


if __name__ == '__main__':
    main()
