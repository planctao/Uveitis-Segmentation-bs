"""Freeze one audited patch-value winner before fresh patient confirmation."""
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


def main():
    out = ROOT/'runs/register_patch_value_confirmation'; assert not (out/'protocol.json').exists()
    source = ROOT/'results/research_register_patch_value'; selection_path = source/'selection_before_repeat.json'
    selected = json.loads(selection_path.read_text())['candidate']
    comparison = json.loads((source/'comparison.json').read_text()); verification = json.loads((source/'verification.json').read_text())
    assert verification['passed'] and verification['actual_seed_heads_checked'] == verification['actual_training_witnesses_checked'] == 48
    assert verification['reused_fusion_source_heads_checked'] == 24 and len(verification['fixed_fusion_phases_checked']) == 16
    eligible = [row for row in comparison['candidates'] if row['development_gate_passed']]
    assert selected == max(eligible, key=lambda row: row['cv']['mean_macro_f1'])
    arm = selected['register_patch_value_configuration']
    assert selected.get('source_register_patch_value_arm', selected['name']) == arm['name']
    weight = selected.get('m8_replacement_weight', 1.)
    assert weight in [.25, 1.]
    registry = json.loads((ROOT/'results/research_reference_registry.json').read_text())
    current_path = ROOT/registry['active_reference']
    assert sha(current_path) == registry['active_reference_sha256']
    current = json.loads(current_path.read_text())
    assert current['retained_development_model'] == 'M8'
    assert selected['cv']['mean_macro_f1'] > current['original_cv_macro_f1']
    assert selected['val']['macro_f1'] >= current['validation_macro_f1']
    assert json.loads((ROOT/'runs/register_patch_value/state.json').read_text())['status'] == 'completed'
    proof_path = ROOT/'verification_register_patch_value_confirmation_v2.json'; proof = json.loads(proof_path.read_text())
    assert proof['passed'] and len(proof['cases']) == 4
    assert proof['candidate'] == selected['candidate_id'] and proof['replacement_weight'] == weight
    for name, expected in proof['code_sha256'].items(): assert sha(ROOT/name) == expected
    for path, expected in proof['input_artifact_sha256'].items(): assert sha(path) == expected
    planning_path = ROOT/'verification_register_patch_value_confirmation_plan_v2.json'
    planning = json.loads(planning_path.read_text())
    assert planning['passed'] and len(planning['plans']) == 6
    assert planning['classification_fits'] == planning['classification_scores'] == planning['external_images_read'] == 0
    for name, expected in planning['code_sha256'].items(): assert sha(ROOT/name) == expected
    parent_path = ROOT/'runs/register_patch_value/protocol.json'; parent = json.loads(parent_path.read_text())
    for key in ['code_sha256', 'frozen_artifacts', 'source_prediction_sha256', 'source_checkpoint_sha256']:
        for name, expected in parent[key].items():
            path = Path(name); assert sha(path if path.is_absolute() else ROOT/path) == expected
    reference = json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())
    raw = reference['selected']['config']; normal = {**raw, 'normal_rank': 32, 'normal_strength': 1.}
    configurations = {'reference': raw, 'subspace': normal,
        'large': {**normal, 'normal_large_feature': 'large_dual'},
        'token': {**normal, 'normal_large_feature': 'large_dual', 'token_evidence_stream': 'registers'}}
    repeats = [20292001, 20292011]; inner_seeds = [20292024+10*ri+fold for ri in range(2) for fold in range(3)]
    old_protocols = sorted(ROOT.glob('runs/*/protocol.json'))
    for path in old_protocols: assert all(str(seed) not in path.read_text() for seed in repeats+inner_seeds), str(path)
    assert shutil.disk_usage('/autodl-fs/data').free > 2*1024**3
    storage = Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124/register_patch_value_confirmation')
    storage.mkdir(parents=True, exist_ok=True)
    if not out.exists(): out.symlink_to(storage, target_is_directory=True)
    assert out.resolve() == storage.resolve()
    frame = pd.read_csv(ROOT/'metadata/manifest.csv'); digest = sha(ROOT/'metadata/manifest.csv')
    original = np.flatnonzero(frame.split.eq('train')); tf = frame.iloc[original]; plans = []
    for ri, repeat in enumerate(repeats):
        splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=repeat); seen = []
        for fold, (inner_local, outer_local) in enumerate(splitter.split(tf, tf.label, tf.split_group)):
            possible = frame.iloc[original[inner_local]]; outer = original[outer_local]
            groups = possible.groupby('split_group').label.agg(lambda values: values.mode().iloc[0])
            inner_seed = 20292024+ri*10+fold
            inner, stopping = train_test_split(groups.index, test_size=.2, random_state=inner_seed, stratify=groups)
            fit = np.zeros(len(frame), bool); stop = fit.copy()
            fit[possible.index[possible.split_group.isin(inner)]] = True
            stop[possible.index[possible.split_group.isin(stopping)]] = True
            check_plan(frame, fit, stop, outer, original); seen.extend(outer.tolist())
            assert frame.loc[fit & frame.label.eq(2)].patient_group.nunique() > 32
            path = out/'split_plans'/f'repeat_{repeat}_fold_{fold}.json'
            dump(path, {'repeat': repeat, 'fold': fold, 'manifest_sha256': digest, 'inner_split_seed': inner_seed,
                'fit_indices': np.flatnonzero(fit).tolist(), 'stop_indices': np.flatnonzero(stop).tolist(), 'outer_indices': outer.tolist()})
            plans.append(path)
        assert sorted(seen) == original.tolist()
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
        'register_patch_value_confirmation.py', 'start_register_patch_value_confirmation_v2.py', 'verify_register_patch_value_confirmation_v2.py',
        'verify_register_patch_value_confirmation_plan_v2.py',
        'normal_large_blend_confirmation.py', 'repeat_patient_cv.py', 'resolution_repeat_cv.py', 'patient_gaussian.py',
        'audit_gaussian_evidence.py', 'repeated_oof.py', 'audit_normal_large.py', 'normal_large_reference_experiments.py',
        'normal_subspace_experiments.py', 'verify_normal_subspace.py', 'large_token_confirmation.py',
        'large_token_evidence_experiments.py', 'audit_large_token_evidence.py'})
    code = sorted(set(code) | {'patch_adapter_cuda_ready.py', 'run_patch_value_confirmation_cuda_ready.py', 'start_patch_value_confirmation_cuda_ready.py'})
    artifacts.add(ROOT/'runs/register_patch_query/audit_cuda_ready_protocol.json')
    protocol = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'manifest_sha256': digest,
        'selection_record': str(selection_path.relative_to(ROOT)), 'selected_candidate': selected['candidate_id'],
        'selected_configuration': {'architecture': arm, 'replacement_weight': weight},
        'configurations': configurations, 'outer_split_seeds': repeats,
        'inner_split_rule': '20292024+10*repeat_index+fold;20% of possible fit patient/near-duplicate groups',
        'fresh_seed_audit': {str(p): sha(p) for p in old_protocols}, 'split_plan_sha256': {str(p): sha(p) for p in plans},
        'arms': ['reference', 'subspace', 'large', 'token', 'value', 'fixed_gaussian', 'm6', 'm7', 'm8', 'candidate'],
        'planned_new_seed_fits': 90, 'planned_generative_fits': 18,
        'hypothesis': 'Confirm the frozen highest qualifying originalCV patch-value candidate. Refit72 rawB/normalB/originalnormalL/registerL sourceheads on sixunused patientplans, then18 valueadapters from the corresponding fresh registerL heads. Allnormalgeometry fit only on fittingpatients and reused exactly by matchingvalue. Keep the original adapter fitting rule and epoch0eligible. Candidate fixed selectedstandalone or .1875rawB+.5625normalB+.25valueL; no confirmation reselection.',
        'reference_comparators': 'M5;fixed.9M5+.1patientGaussian;M6=.25rawB+.75normalB;M7=.1875rawB+.5625normalB+.25normalL;M8=.1875rawB+.5625normalB+.25registerL.',
        'confirmation_gate': 'Mean sixfold MacroF1 strictly higher AND paired patient95% lower endpoint>0 versus EACH M8/M7/M6/M5/fixedGaussian. No automatic external evaluation.',
        'external_test_policy': 'Only original1717 trainingrows form fit/stop/outer; validation unused; all externalfeature rows NaN. No external outcomes or predictions loaded.',
        'audit': 'Freeze all6plans/18geometries before90supervisedtrajectories. Freeze eachfresh registerL source and verify zerovalue identity before itsadapter. Freeze allproducts beforeindependentaudit. Restore all90selectedheads/stats/maps/epochs/stop/outerprobs, all54actualCUDA normalL/registerL/query complete tasks/alltrainablegradients,18Gaussianmoments, tenensembles/fivepairedcomparisons. Independentfloat64 fixedfusion arithmetic. All original code/bank/tail/reference immutable.',
        'frozen_artifacts': {str(p): sha(p) for p in sorted(artifacts)}, 'code_sha256': {name: sha(ROOT/name) for name in code}}
    protocol['execution_entrypoint'] = 'run_patch_value_confirmation_cuda_ready.py'
    protocol['cuda_restore_recovery'] = 'CUDA model conversion outside inference_mode; originaltrained tensors, fitting code and strictchecks unchanged.'
    dump(out/'protocol.json', protocol)
    dump(out/'state.json', {'status': 'frozen_before_training', 'new_seed_fits': 0, 'external_test_evaluated': False})
    env = dict(os.environ); env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    with (ROOT/'register_patch_value_confirmation.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT/'run_patch_value_confirmation_cuda_ready.py')], cwd=ROOT,
            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    hypothesis_path = ROOT/'results/next_register_patch_value_hypothesis.json'; hypothesis = json.loads(hypothesis_path.read_text())
    dump(hypothesis_path, {**hypothesis, 'status': 'fresh_patient_confirmation_running', 'selected_candidate': selected['candidate_id'],
        'confirmation_protocol': str((out/'protocol.json').relative_to(ROOT)), 'confirmation_protocol_sha256': sha(out/'protocol.json'), 'confirmation_pid': process.pid})
    print(json.dumps({'pid': process.pid, 'protocol_sha256': sha(out/'protocol.json'), 'planned_new_seed_fits': 90, 'outer_split_seeds': repeats}), flush=True)


if __name__ == '__main__': main()
