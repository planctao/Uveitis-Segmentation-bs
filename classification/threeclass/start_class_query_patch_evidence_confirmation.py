"""Freeze the highest eligible direct patch-evidence candidate before fresh patient confirmation."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from experiment import ROOT, dump
from kernel_residual_experiments import sha
from resolution_repeat_cv import check_plan
from class_query_patch_evidence_plans import make_plans, assert_unused, OUTER_SEEDS


def main():
    out = ROOT/'runs/class_query_patch_evidence_confirmation'; assert not (out/'protocol.json').exists()
    source = ROOT/'results/research_class_query_patch_evidence'; selection_path = source/'selection_before_repeat.json'
    selected = json.loads(selection_path.read_text())['candidate']
    comparison = json.loads((source/'comparison.json').read_text()); verification = json.loads((source/'verification.json').read_text())
    assert verification['passed'] and verification['actual_seed_heads_checked'] == verification['actual_training_witnesses_checked'] == 48
    assert verification['reused_fusion_source_heads_checked'] == 24 and len(verification['fixed_fusion_phases_checked']) == 16
    eligible = [row for row in comparison['candidates'] if row['development_gate_passed']]
    assert selected == max(eligible, key=lambda row: row['cv']['mean_macro_f1'])
    arm = selected['class_query_patch_evidence_configuration']
    assert selected.get('source_class_query_patch_evidence_arm', selected['name']) == arm['name']
    weight = selected.get('m8_replacement_weight', 1.)
    assert weight in [.25, 1.]
    registry = json.loads((ROOT/'results/research_reference_registry.json').read_text())
    current_path = ROOT/registry['active_reference']
    assert sha(current_path) == registry['active_reference_sha256']
    current = json.loads(current_path.read_text())
    assert current['retained_development_model'] == 'M8'
    assert selected['cv']['mean_macro_f1'] > current['original_cv_macro_f1']
    assert selected['val']['macro_f1'] >= current['validation_macro_f1']
    assert json.loads((ROOT/'runs/class_query_patch_evidence/state.json').read_text())['status'] == 'completed'
    proof_path = ROOT/'verification_class_query_patch_evidence_confirmation.json'; proof = json.loads(proof_path.read_text())
    assert proof['passed'] and len(proof['cases']) == 4
    assert proof['candidate'] == selected['candidate_id'] and proof['replacement_weight'] == weight
    for name, expected in proof['code_sha256'].items(): assert sha(ROOT/name) == expected
    for path, expected in proof['input_artifact_sha256'].items(): assert sha(path) == expected
    planning_path = ROOT/'verification_class_query_patch_evidence_confirmation_plan.json'
    planning = json.loads(planning_path.read_text())
    assert planning['passed'] and len(planning['plans']) == 6
    assert planning['classification_fits'] == planning['classification_scores'] == planning['external_images_read'] == 0
    for name, expected in planning['code_sha256'].items(): assert sha(ROOT/name) == expected
    assert len(planning['synthetic_formula_cases'])==4 and len(planning['actual_matched_control_engine_cases'])==2
    assert planning['learned_winner_supervised_fits']==126 and planning['learned_winner_training_witnesses']==90
    assert planning['control_winner_supervised_fits']==90 and planning['control_winner_training_witnesses']==54
    assert len(planning['development_gate_cases'])==40 and len(planning['confirmation_gate_cases'])==24
    assert len(planning['actual_prediction_restore_engine_cases'])==4
    parent_path = ROOT/'runs/class_query_patch_evidence/protocol.json'; parent = json.loads(parent_path.read_text())
    for key in ['code_sha256', 'frozen_artifacts', 'source_prediction_sha256', 'source_checkpoint_sha256']:
        for name, expected in parent[key].items():
            path = Path(name); assert sha(path if path.is_absolute() else ROOT/path) == expected
    reference = json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())
    raw = reference['selected']['config']; normal = {**raw, 'normal_rank': 32, 'normal_strength': 1.}
    configurations = {'reference': raw, 'subspace': normal,
        'large': {**normal, 'normal_large_feature': 'large_dual'},
        'token': {**normal, 'normal_large_feature': 'large_dual', 'token_evidence_stream': 'registers'}}
    assert arm['pooling'] in ['uniform','source','learned','contrast']
    coupled = arm['learned_query']
    planned_fits=126 if coupled else 90
    planned_witnesses=90 if coupled else 54
    repeats = OUTER_SEEDS
    old_protocols = assert_unused(ROOT)
    assert shutil.disk_usage('/autodl-fs/data').free > 2*1024**3
    storage = Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124/class_query_patch_evidence_confirmation')
    storage.mkdir(parents=True, exist_ok=True)
    if not out.exists(): out.symlink_to(storage, target_is_directory=True)
    assert out.resolve() == storage.resolve()
    frame = pd.read_csv(ROOT/'metadata/manifest.csv'); digest = sha(ROOT/'metadata/manifest.csv')
    plans=[]
    for plan in make_plans(frame,digest):
        path=out/'split_plans'/f"repeat_{plan['repeat']}_fold_{plan['fold']}.json"
        dump(path,plan);plans.append(path)
    artifacts = {Path(path) for path in parent['frozen_artifacts']}
    artifacts.update([proof_path, planning_path, selection_path, parent_path, current_path, ROOT/'results/current_research_reference.json',
        ROOT/'features/multiscale_vit.npz', ROOT/'features/multiscale_vit.json', ROOT/'verification_large_token_evidence.json'])
    artifacts.update(source/name for name in ['comparison.json', 'verification.json', 'freeze_integrity_after_training.json'])
    for key in ['package', 'selection', 'inference_verification']:
        path = ROOT/current[key]; assert sha(path) == current[key+'_sha256']; artifacts.add(path)
    selected_record = json.loads(selection_path.read_text())
    for key in ['checkpoint_sha256']:
        for path, expected in selected_record[key].items(): assert sha(path) == expected; artifacts.add(Path(path))
    code = sorted(set(parent['code_sha256']) | {
        'class_query_patch_evidence_confirmation.py', 'start_class_query_patch_evidence_confirmation.py', 'verify_class_query_patch_evidence_confirmation.py',
        'verify_class_query_patch_evidence_confirmation_plan.py', 'class_query_patch_evidence_plans.py',
        'normal_large_blend_confirmation.py', 'repeat_patient_cv.py', 'resolution_repeat_cv.py', 'patient_gaussian.py',
        'audit_gaussian_evidence.py', 'repeated_oof.py', 'audit_normal_large.py', 'normal_large_reference_experiments.py',
        'normal_subspace_experiments.py', 'verify_normal_subspace.py', 'large_token_confirmation.py',
        'large_token_evidence_experiments.py', 'audit_large_token_evidence.py'})
    from start_class_query_patch_evidence import scientific_closure
    code=scientific_closure(code)
    protocol = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'manifest_sha256': digest,
        'selection_record': str(selection_path.relative_to(ROOT)), 'selected_candidate': selected['candidate_id'],
        'selected_configuration': {'architecture': arm, 'replacement_weight': weight},
        'configurations': configurations, 'outer_split_seeds': repeats,
        'inner_split_rule': '20293624+10*repeat_index+fold;20% of possible fit patient/near-duplicate groups',
        'fresh_seed_audit': {str(p): sha(p) for p in old_protocols}, 'split_plan_sha256': {str(p): sha(p) for p in plans},
        'arms': ['reference', 'subspace', 'large', 'token', 'value', 'fixed_gaussian', 'm6', 'm7', 'm8', 'candidate']+(['matched_uniform','matched_uniform_control','matched_source_query','matched_source_query_control'] if coupled else []),
        'planned_new_seed_fits': planned_fits, 'planned_training_witnesses': planned_witnesses, 'planned_generative_fits': 18,
        'hypothesis': 'Freeze the highest eligible originalCV direct patch-evidence candidate once. Fit72 rawB/normalB/normalL/registerL heads, then18 selected candidate continuations from matching fresh registerL sources. Coupled winners additionally fit18 uniform and18 source-query controls. Control winner90fits/54witnesses; coupled126/90. No confirmation reselection.',
        'reference_comparators': 'M5;fixed.9M5+.1patientGaussian;M6=.25rawB+.75normalB;M7=.1875rawB+.5625normalB+.25normalL;M8=.1875rawB+.5625normalB+.25registerL.',
        'matched_control': 'For coupled winners use fresh uniform and source-query controls, same source/init/order/task/training rule and selected standalone/fusion weight. Different trainable parameter counts; no matched-total-budget claim.',
        'confirmation_gate': 'Higher mean sixfold MacroF1 and paired patient95% lower endpoint>0 versus EACH M8/M7/M6/M5/fixedGaussian and both matched controls if coupled. No automatic external evaluation or promotion.',
        'external_test_policy': 'Only original1717 trainingrows form fit/stop/outer; validation unused; all externalfeature rows NaN. No external outcomes or predictions loaded.',
        'audit': 'Freeze all6plans/18geometries beforeallplannedtrajectories. Freeze eachfresh registerL source and verify zerovalue identity before itsadapter. Freeze allproducts beforeindependentaudit. Restore allplannedselectedheads/stats/maps/epochs/stop/outerprobs, allplannedactualCUDA normalL/registerL/direct patch-evidence/control complete tasks/alltrainablegradients,18Gaussianmoments, all14/10declaredensembles/sevenorfivepairedcomparisons. Independentfloat64 fixedfusion arithmetic. All original code/bank/tail/reference immutable.',
        'execution_entrypoint': 'class_query_patch_evidence_confirmation.py',
        'frozen_artifacts': {str(p): sha(p) for p in sorted(artifacts)}, 'code_sha256': {name: sha(ROOT/name) for name in code}}
    dump(out/'protocol.json', protocol)
    dump(out/'state.json', {'status': 'frozen_before_training', 'new_seed_fits': 0, 'external_test_evaluated': False})
    env = dict(os.environ); env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    with (ROOT/'class_query_patch_evidence_confirmation.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT/'class_query_patch_evidence_confirmation.py')], cwd=ROOT,
            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    hypothesis_path = ROOT/'results/next_class_query_patch_evidence_hypothesis.json'; hypothesis = json.loads(hypothesis_path.read_text())
    dump(hypothesis_path, {**hypothesis, 'status': 'fresh_patient_confirmation_running', 'selected_candidate': selected['candidate_id'],
        'confirmation_protocol': str((out/'protocol.json').relative_to(ROOT)), 'confirmation_protocol_sha256': sha(out/'protocol.json'), 'confirmation_pid': process.pid})
    print(json.dumps({'pid': process.pid, 'protocol_sha256': sha(out/'protocol.json'), 'planned_new_seed_fits': planned_fits, 'planned_training_witnesses': planned_witnesses, 'outer_split_seeds': repeats}), flush=True)


if __name__ == '__main__': main()
