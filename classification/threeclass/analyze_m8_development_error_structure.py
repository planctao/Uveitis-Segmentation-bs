"""Describe existing M8 training OOF errors without fitting or external scoring."""
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from experiment import ROOT, dump, metrics
from kernel_residual_experiments import sha


def main():
    output = ROOT/'results/m8_development_error_structure'
    assert not (output/'analysis.json').exists()
    frame = pd.read_csv(ROOT/'metadata/manifest.csv')
    reference_path = ROOT/'results/research_reference_m8.json'
    reference = json.loads(reference_path.read_text())
    assert reference['retained_development_model'] == 'M8'
    inputs = {str(reference_path): sha(reference_path), str(ROOT/'metadata/manifest.csv'): sha(ROOT/'metadata/manifest.csv')}
    rows, fold_metrics = [], []
    for fold in range(3):
        phase = f'fold_{fold}'
        paths = {'rawB': ROOT/'runs/normal_subspace/zero_control'/phase/'predictions.npz',
            'normalB': ROOT/'runs/normal_subspace/normal_rank32_full'/phase/'predictions.npz',
            'registerL': ROOT/'runs/large_token_evidence/registers'/phase/'predictions.npz',
            'm8': ROOT/'runs/large_token_evidence/registers_m7replace'/phase/'predictions.npz'}
        banks = {name: dict(np.load(path)) for name, path in paths.items()}
        inputs.update({str(path): sha(path) for path in paths.values()})
        ids = banks['m8']['indices']
        assert all(np.array_equal(ids, bank['indices']) for bank in banks.values())
        assert frame.iloc[ids].split.eq('train').all()
        expected = .1875*banks['rawB']['p']+.5625*banks['normalB']['p']+.25*banks['registerL']['p']
        probability = banks['m8']['p']
        assert np.allclose(probability, expected, atol=2e-7, rtol=2e-7)
        assert np.array_equal(probability.argmax(1), expected.argmax(1))
        fold_metrics.append({'fold': fold, 'metrics': metrics(frame.iloc[ids].label, probability)})
        decisions = {name: bank['p'].argmax(1) for name, bank in banks.items()}
        for local, index in enumerate(ids):
            source = frame.iloc[index]
            y = int(source.label)
            ranking = np.sort(probability[local])
            rows.append({'manifest_index': int(index), 'fold': fold, 'disease': source.disease,
                'patient_group': source.patient_group, 'split_group': source.split_group,
                'true_label': y, 'predicted_label': int(decisions['m8'][local]),
                'error': int(decisions['m8'][local] != y),
                'rawB_prediction': int(decisions['rawB'][local]),
                'normalB_prediction': int(decisions['normalB'][local]),
                'registerL_prediction': int(decisions['registerL'][local]),
                'B_L_disagree': bool(decisions['normalB'][local] != decisions['registerL'][local]),
                'B_L_agree_wrong': bool(decisions['normalB'][local] == decisions['registerL'][local] != y),
                'm8_probability_true_label': float(probability[local, y]),
                'm8_top_two_margin': float(ranking[-1]-ranking[-2])})
    mean_cv = float(np.mean([row['metrics']['macro_f1'] for row in fold_metrics]))
    assert abs(mean_cv-reference['original_cv_macro_f1']) < 1e-12
    data = pd.DataFrame(rows).sort_values('manifest_index')
    assert data.manifest_index.is_unique
    assert data.manifest_index.tolist() == np.flatnonzero(frame.split.eq('train')).tolist()
    disease_rows = []
    for disease, group in data.groupby('disease'):
        assert group.true_label.nunique() == 1
        disease_rows.append({'disease': disease, 'true_label': int(group.true_label.iloc[0]),
            'images': len(group), 'patients': group.patient_group.nunique(),
            'patients_with_any_error': group.loc[group.error.eq(1)].patient_group.nunique(),
            'errors': int(group.error.sum()),
            'prediction_counts': {str(label): int(group.predicted_label.eq(label).sum()) for label in range(3)},
            'B_L_agree_wrong': int(group.B_L_agree_wrong.sum()),
            'B_L_disagree': int(group.B_L_disagree.sum())})
    disease_rows.sort(key=lambda row: (-row['errors'], row['disease']))
    disagreement_rows = []
    for name, selected in [('B_L_agree', ~data.B_L_disagree), ('B_L_disagree', data.B_L_disagree),
        ('B_L_agree_wrong', data.B_L_agree_wrong)]:
        subset = data.loc[selected]
        disagreement_rows.append({'subset': name, 'images': len(subset),
            'patients': subset.patient_group.nunique(), 'm8_errors': int(subset.error.sum())})
    confusion = np.zeros((3, 3), dtype=int)
    for row in rows:
        confusion[row['true_label'], row['predicted_label']] += 1
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output/'training_oof_cases.csv', index=False)
    dump(output/'analysis.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Descriptive diagnosis from existing selected M8 training OOF probabilities. These original folds were already used in model selection; this is not fresh confirmation or a new improvement claim.',
        'scope': 'Exactly original1717training rows, eachonce; validation and external rows excluded.',
        'new_model_fits': 0, 'new_candidate_models_scored': 0, 'external_test_evaluated': False,
        'original_cv_mean_macro_f1_restored': mean_cv, 'fold_metrics': fold_metrics,
        'confusion_matrix': confusion.tolist(), 'total_training_oof_errors': int(data.error.sum()),
        'disease_error_counts': disease_rows, 'branch_agreement_error_counts': disagreement_rows,
        'limitation': 'Image error counts are descriptive and correlated within patients. Rare subtype counts do not establish reliable per-disease effects. The same broad subtype label can include different acquisition contexts; do not infer a causal imaging mechanism.',
        'input_artifact_sha256': inputs, 'case_table_sha256': sha(output/'training_oof_cases.csv'),
        'code_sha256': sha(ROOT/'analyze_m8_development_error_structure.py')})
    print(json.dumps({'errors': int(data.error.sum()), 'images': len(data), 'confusion': confusion.tolist(),
        'largest_error_counts': disease_rows[:6], 'branch_agreement': disagreement_rows}, ensure_ascii=False))


if __name__ == '__main__':
    main()
