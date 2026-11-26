"""Describe completed mass-calibrated OOF corrections and fixed-fusion retention."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from experiment import ROOT, metrics, dump

DEST = ROOT/'results/mass_calibration_correction_structure'
PROTOCOL = ROOT/'runs/mass_calibration_correction_diagnostic/protocol.json'
FAMILIES = ['register_patch_mass_calibration']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def events(y, before, after, patients):
    previous, current = before.argmax(1), after.argmax(1)
    corrected = (previous != y) & (current == y)
    harmed = (previous == y) & (current != y)
    changed = previous != current
    return {
        'changed_decisions': int(changed.sum()), 'corrected_errors': int(corrected.sum()),
        'introduced_errors': int(harmed.sum()), 'net_correct_change': int(corrected.sum()-harmed.sum()),
        'changed_decision_patients': int(pd.Series(patients[changed]).nunique()),
        'corrected_error_patients': int(pd.Series(patients[corrected]).nunique()),
        'introduced_error_patients': int(pd.Series(patients[harmed]).nunique()),
        'maximum_absolute_probability_change': float(np.abs(after-before).max()),
    }, corrected, harmed


def main():
    assert not (DEST/'analysis.json').exists()
    protocol = json.loads(PROTOCOL.read_text())
    assert sha(__file__) == protocol['code_sha256']
    for name,expected in protocol['input_sha256'].items():
        assert sha(name) == expected, name
    frame = pd.read_csv(ROOT/'metadata/manifest.csv')
    assert sha(ROOT/'metadata/manifest.csv') == '2778cbc34db8d588770f036d0f1dba728cad631634697556a88e7daf25ea8c02'
    inputs = dict(protocol['input_sha256'])
    inputs[str(PROTOCOL)] = sha(PROTOCOL)

    def read(path):
        path = Path(path)
        assert inputs[str(path)] == sha(path), str(path)
        with np.load(path) as saved:
            result = {name: saved[name].copy() for name in saved.files}
        assert np.array_equal(result['pred'], result['p'].argmax(1))
        return result

    original = {}
    for fold in range(3):
        phase = f'fold_{fold}'
        original[phase] = {
            name: read(ROOT/'runs'/folder/phase/'predictions.npz')
            for name, folder in {
                'rawB': 'normal_subspace/zero_control',
                'normalB': 'normal_subspace/normal_rank32_full',
                'L': 'large_token_evidence/registers',
                'M8': 'large_token_evidence/registers_m7replace',
            }.items()}
        indices = original[phase]['M8']['indices']
        assert frame.iloc[indices].split.eq('train').all()
        assert all(np.array_equal(row['indices'], indices) for row in original[phase].values())
        expected = .1875*original[phase]['rawB']['p']+.5625*original[phase]['normalB']['p']+.25*original[phase]['L']['p']
        assert np.allclose(expected, original[phase]['M8']['p'], atol=2e-7, rtol=2e-7)
    candidate_rows, case_rows = [], []
    for family in FAMILIES:
        run = ROOT/'runs'/family
        result = ROOT/'results'/('research_'+family)
        state = json.loads((run/'state.json').read_text())
        verification = json.loads((result/'verification.json').read_text())
        comparison = json.loads((result/'comparison.json').read_text())
        assert state['status'] == 'completed' and state['new_seed_fits'] == 48
        assert verification['passed'] and verification['actual_seed_heads_checked'] == 48
        assert comparison['new_seed_fits'] == 48 and not comparison['external_test_evaluated']
        for path in [run/'protocol.json', result/'comparison.json', result/'verification.json']:
            inputs[str(path)] = sha(path)
        compared = {row['name']: row for row in comparison['candidates']}
        arms = [row['name'] for row in json.loads((run/'protocol.json').read_text())['configuration_plan']]
        assert len(arms) == 4 and len(compared) == 8
        for arm in arms:
            pieces = []
            for fold in range(3):
                phase = f'fold_{fold}'
                source = original[phase]
                adapted = read(run/arm/phase/'predictions.npz')
                fused = read(run/(arm+'_m8replace')/phase/'predictions.npz')
                ids = source['M8']['indices']
                assert np.array_equal(ids, adapted['indices']) and np.array_equal(ids, fused['indices'])
                expected = (.1875*source['rawB']['p']+.5625*source['normalB']['p']+.25*adapted['p']).astype('float32')
                assert np.array_equal(expected,fused['p'])
                y = frame.iloc[ids].label.to_numpy()
                pieces.append({'indices': ids, 'L': source['L']['p'], 'M8': source['M8']['p'],
                    'adapted': adapted['p'], 'fused': fused['p']})
                for name, bank in [(arm, adapted), (arm+'_m8replace', fused)]:
                    actual = metrics(y, bank['p'])
                    expected_metric = compared[name]['cv']['folds'][fold]['metrics']
                    assert abs(actual['macro_f1']-expected_metric['macro_f1']) < 1e-12
            all_ids = np.concatenate([row['indices'] for row in pieces])
            assert np.array_equal(np.sort(all_ids), np.flatnonzero(frame.split.eq('train')))
            banks = {name: np.concatenate([row[name] for row in pieces]) for name in ['L', 'M8', 'adapted', 'fused']}
            y = frame.iloc[all_ids].label.to_numpy()
            patients = frame.iloc[all_ids].patient_group.to_numpy()
            standalone, corrected_l, harmed_l = events(y, banks['L'], banks['adapted'], patients)
            fusion, corrected_m8, harmed_m8 = events(y, banks['M8'], banks['fused'], patients)
            standalone.update({
                'corrected_L_errors_where_M8_was_already_correct': int((corrected_l & (banks['M8'].argmax(1) == y)).sum()),
                'corrected_L_errors_where_M8_was_wrong': int((corrected_l & (banks['M8'].argmax(1) != y)).sum()),
                'corrected_L_errors_also_fixing_M8_after_fusion': int((corrected_l & corrected_m8).sum()),
            })
            row = {'family': family, 'arm': arm, 'original_training_oof_images': len(y),
                'standalone_vs_source_L': standalone, 'fixed_fusion_vs_M8': fusion,
                'original_predeclared_fused_cv_macro_f1': compared[arm+'_m8replace']['cv']['mean_macro_f1'],
                'original_predeclared_fused_validation_macro_f1': compared[arm+'_m8replace']['val']['macro_f1'],
                'development_gate_passed': compared[arm+'_m8replace']['development_gate_passed']}
            control_name = arm.replace('source_mass_','constant_mass_')
            control_pieces = [read(run/control_name/f'fold_{fold}'/'predictions.npz') for fold in range(3)]
            control_fused_pieces = [read(run/(control_name+'_m8replace')/f'fold_{fold}'/'predictions.npz') for fold in range(3)]
            assert all(np.array_equal(x['indices'],pieces[fold]['indices']) for fold,x in enumerate(control_pieces))
            assert all(np.array_equal(x['indices'],pieces[fold]['indices']) for fold,x in enumerate(control_fused_pieces))
            control_p = np.concatenate([x['p'] for x in control_pieces])
            control_fused = np.concatenate([x['p'] for x in control_fused_pieces])
            row['same_kind_constant_control'] = control_name
            row['standalone_vs_same_kind_constant'] = events(y,control_p,banks['adapted'],patients)[0]
            row['fusion_vs_same_kind_constant'] = events(y,control_fused,banks['fused'],patients)[0]
            candidate_rows.append(row)
            for local, index in enumerate(all_ids):
                case_rows.append({'family': family, 'arm': arm, 'manifest_index': int(index),
                    'patient_group': patients[local], 'true_label': int(y[local]),
                    'source_L_prediction': int(banks['L'][local].argmax()),
                    'adapted_L_prediction': int(banks['adapted'][local].argmax()),
                    'M8_prediction': int(banks['M8'][local].argmax()),
                    'adapted_fusion_prediction': int(banks['fused'][local].argmax()),
                    'same_kind_constant_prediction':int(control_p[local].argmax()),
                    'same_kind_constant_fusion_prediction':int(control_fused[local].argmax()),
                    'L_error_corrected': bool(corrected_l[local]), 'L_error_introduced': bool(harmed_l[local]),
                    'M8_error_corrected': bool(corrected_m8[local]), 'M8_error_introduced': bool(harmed_m8[local])})
    assert len(candidate_rows) == 4 and len(case_rows) == 4*1717
    for path, expected in inputs.items():
        assert sha(path) == expected, path
    DEST.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(case_rows).sort_values(['family', 'arm', 'manifest_index']).to_csv(DEST/'training_oof_changes.csv', index=False)
    dump(DEST/'analysis.json', {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Post-outcome descriptive analysis of existing selected mass-calibration original-training OOF decisions, fixed M8 fusion retention and paired same-kind fit-mean control differences. No new candidate, weight or threshold search.',
        'scope': 'Exactly1717originaltrainingOOFimages per each of4completedmasscalibrationarms; validationmetrics copied only from existing frozencomparison, no validationpredictions/externaloutputs read.',
        'limitations': 'Original OOF folds already participated in model selection. Counts are correlated within patients and across arms; do not sum across arms as independent outcomes. Standalone baseline is sourceL whereas fused baseline is M8, so correction counts have different starting errors. Descriptive correction retention does not prove a causal fusion bottleneck or establish that a different weight would help.',
        'new_model_fits': 0, 'new_model_predictions': 0, 'new_candidate_scores': 0,
        'external_test_evaluated': False, 'rows': candidate_rows,
        'input_sha256': inputs, 'case_table_sha256': sha(DEST/'training_oof_changes.csv'),
        'code_sha256': sha(__file__), 'protocol_sha256': sha(PROTOCOL)})
    print(json.dumps(candidate_rows), flush=True)


if __name__ == '__main__':
    main()
