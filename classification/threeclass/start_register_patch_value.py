"""Freeze queued value adaptation only after query study and any confirmation end."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from experiment import ROOT, dump
from kernel_residual_experiments import sha


def main():
    out = ROOT/'runs/register_patch_value'
    assert not (out/'protocol.json').exists()
    source = ROOT/'results/research_register_patch_query'
    state = json.loads((ROOT/'runs/register_patch_query/state.json').read_text())
    assert state['status'] == 'completed'
    verification = json.loads((source/'verification.json').read_text())
    comparison = json.loads((source/'comparison.json').read_text())
    assert verification['passed']
    assert verification['actual_seed_heads_checked'] == verification['actual_training_witnesses_checked'] == 48
    assert verification['reused_fusion_source_heads_checked'] == 24 and len(verification['fixed_fusion_phases_checked']) == 16
    eligible = [row for row in comparison['candidates'] if row['development_gate_passed']]
    assert state['passing_development_gates'] == len(eligible)
    confirmation_artifacts = []
    if eligible:
        selected = json.loads((source/'selection_before_repeat.json').read_text())['candidate']
        assert selected == max(eligible, key=lambda row: row['cv']['mean_macro_f1'])
        confirmation = ROOT/'results/research_register_patch_query_confirmation'
        confirm_state = json.loads((ROOT/'runs/register_patch_query_confirmation/state.json').read_text())
        confirm_result = json.loads((confirmation/'comparison.json').read_text())
        confirm_audit = json.loads((confirmation/'verification.json').read_text())
        assert confirm_state['status'] == 'completed' and confirm_audit['passed']
        assert confirm_audit['actual_seed_heads_checked'] == 90 and confirm_audit['actual_training_witnesses_checked'] == 54
        assert not confirm_result['confirmation_gate_passed'], 'A passing confirmation requires reference review before this queued family.'
        confirmation_artifacts = [confirmation/name for name in ['comparison.json', 'verification.json', 'freeze_integrity_after_confirmation.json']]
    registry = json.loads((ROOT/'results/research_reference_registry.json').read_text())
    reference_path = ROOT/registry['active_reference']
    assert sha(reference_path) == registry['active_reference_sha256']
    current = json.loads(reference_path.read_text())
    assert current['retained_development_model'] == 'M8'
    bank_path = ROOT/'verification_register_patch_query_bank.json'
    proof_path = ROOT/'verification_register_patch_value_model.json'
    prefix_path = ROOT/'verification_register_patch_value_prefix256.json'
    for path in [bank_path, proof_path, prefix_path]:
        proof = json.loads(path.read_text())
        assert proof['passed'] and proof['classification_fits'] == proof['classification_scores'] == proof['external_images_read'] == 0
    proof = json.loads(proof_path.read_text())
    assert len(proof['cases']) == 4 and all(case['actual_images'] == 256 for case in proof['cases'])
    assert proof['bank_verification_sha256'] == sha(bank_path)
    for name, expected in proof['code_sha256'].items(): assert sha(ROOT/name) == expected
    prefix = json.loads(prefix_path.read_text())
    for name, expected in prefix['code_sha256'].items(): assert sha(ROOT/name) == expected
    parent_path = ROOT/'runs/register_patch_query/protocol.json'
    parent = json.loads(parent_path.read_text())
    for section in ['code_sha256', 'frozen_artifacts', 'source_checkpoint_sha256', 'source_prediction_sha256']:
        for name, expected in parent[section].items():
            path = Path(name)
            assert sha(path if path.is_absolute() else ROOT/path) == expected, name
    hypothesis = ROOT/'results/next_register_patch_value_hypothesis.json'
    planned = json.loads(hypothesis.read_text())
    assert planned['classification_fits'] == planned['scores_evaluated'] == 0
    artifacts = {Path(name) for name in parent['frozen_artifacts']}
    artifacts.update([parent_path, bank_path, proof_path, prefix_path, reference_path])
    artifacts.update(confirmation_artifacts)
    artifacts.update(source/name for name in ['comparison.json', 'verification.json', 'freeze_integrity_after_training.json'])
    code = sorted(set(parent['code_sha256']) | {
        'register_patch_value.py', 'audit_register_patch_value.py', 'register_patch_value_witness.py',
        'register_patch_value_experiments.py', 'verify_register_patch_value_prefix.py',
        'verify_register_patch_value_model.py', 'start_register_patch_value.py'})
    assert shutil.disk_usage('/autodl-fs/data').free > 1024**3
    storage = Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124/register_patch_value')
    storage.mkdir(parents=True, exist_ok=True)
    if not out.exists(): out.symlink_to(storage, target_is_directory=True)
    assert out.resolve() == storage.resolve()
    protocol = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'manifest_sha256': sha(ROOT/'metadata/manifest.csv'),
        'retained_model': 'M8', 'retained_reference': str(reference_path.relative_to(ROOT)),
        'minimum_cv_macro_f1': current['original_cv_macro_f1'], 'minimum_validation_macro_f1': current['validation_macro_f1'],
        'hypothesis': planned['hypothesis'], 'architecture': planned['architecture_plan'],
        'configuration_plan': planned['configuration_plan'], 'planned_new_seed_fits': 48, 'predeclared_scores': 8,
        'training': planned['fixed_training'], 'fixed_fusions': planned['fixed_fusions'], 'selection': planned['selection'],
        'external_test_policy': 'Development2091 bank only; original1717train/374validation; no externalimage or externalprediction reads.',
        'source_checkpoint_sha256': parent['source_checkpoint_sha256'], 'source_prediction_sha256': parent['source_prediction_sha256'],
        'code_sha256': {name: sha(ROOT/name) for name in code}, 'frozen_artifacts': {str(path): sha(path) for path in sorted(artifacts)},
        'actual_fitting_count_policy': '48 new supervised valueadapter classifierfits. Reuse/audit12sourceM8L and24B heads without counting again.'}
    dump(out/'protocol.json', protocol)
    dump(out/'state.json', {'status': 'frozen_before_training', 'new_seed_fits': 0, 'external_test_evaluated': False})
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    with (ROOT/'register_patch_value.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT/'register_patch_value_experiments.py')], cwd=ROOT,
            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    dump(hypothesis, {**planned, 'status': 'running', 'protocol': 'runs/register_patch_value/protocol.json',
        'protocol_sha256': sha(out/'protocol.json'), 'pid': process.pid, 'model_preflight': str(proof_path.relative_to(ROOT))})
    print(json.dumps({'pid': process.pid, 'protocol_sha256': sha(out/'protocol.json'), 'planned_new_seed_fits': 48}), flush=True)


if __name__ == '__main__': main()
