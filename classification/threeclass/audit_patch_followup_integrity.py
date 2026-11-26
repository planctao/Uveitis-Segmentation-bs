"""Supplement a finished global audit with later patch studies and continuations.

This verifies frozen bytes and completed study accounting, not fresh predictions.
The prior global result remains an immutable, explicitly scoped base record.
"""
import fcntl
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import traceback
from experiment import ROOT, dump
from kernel_residual_experiments import sha

OUT = ROOT/'runs/patch_followup_integrity'
DEST = ROOT/'results/patch_followup_integrity.json'


def fingerprint(path):
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def run():
    protocol = json.loads((OUT/'protocol.json').read_text())
    checked, physical, targets = {}, {}, {}
    def check(name, expected):
        path = Path(name)
        path = path if path.is_absolute() else ROOT/path
        target = path.resolve(strict=True)
        before = fingerprint(target)
        previous = physical.get(str(target))
        if previous is None:
            actual = sha(target)
            assert fingerprint(target) == before, ('Changed while hashing', str(path))
            physical[str(target)] = {'fingerprint': before, 'sha256': actual}
            if len(physical) % 100 == 0:
                print(json.dumps({'event': 'followup_integrity_progress', 'physical_files': len(physical)}), flush=True)
        else:
            assert previous['fingerprint'] == before, ('Changed between uses', str(path))
            actual = previous['sha256']
        assert actual == expected, (str(path), actual, expected)
        checked[str(path)] = actual
        targets[str(path)] = str(target)
    for key in ['code_sha256', 'frozen_artifacts']:
        for name, expected in protocol[key].items():
            check(name, expected)
    base_path = ROOT/protocol['base_global_audit_archive']
    base = json.loads(base_path.read_text())
    assert base['passed'] and 'register_patch_query' in base['studies']
    assert 'register_patch_value' not in base['studies'], 'This supplement targets a base collected before Value completed.'
    assert base['tracked_fully_audited_heads_total'] == protocol['tracked_count_at_freeze']
    completed, active_protocols = {}, []
    for family in ['register_patch_value', 'register_patch_joint', 'register_patch_gated_value',
        'register_patch_readout_adaptation', 'inflammatory_pair_specialist']:
        folder = ROOT/'runs'/family
        if not (folder/'protocol.json').exists():
            continue
        formal = json.loads((folder/'protocol.json').read_text())
        check(folder/'protocol.json', sha(folder/'protocol.json'))
        for key in ['code_sha256', 'frozen_artifacts', 'source_checkpoint_sha256', 'source_prediction_sha256']:
            for name, expected in formal[key].items():
                check(name, expected)
        geometry = folder/'geometry_freeze_before_all_fits.json'
        if geometry.exists():
            frozen = json.loads(geometry.read_text())
            assert len(frozen['source_geometry_sha256']) == len(frozen['split_plan_sha256']) == 4
            for key in ['source_geometry_sha256', 'split_plan_sha256']:
                for name, expected in frozen[key].items():
                    check(name, expected)
        state = json.loads((folder/'state.json').read_text())
        if state['status'] != 'completed':
            cmd = Path('/proc')/str(state.get('pid'))/'cmdline'
            command = cmd.read_bytes().split(b'\0') if cmd.exists() else []
            assert str(ROOT/formal['execution_entrypoint']).encode() in command, (family, state)
            active_protocols.append({'family': family, 'state_at_inspection': state,
                'training_outputs_excluded_from_frozen_byte_check': True})
            continue
        result = ROOT/'results'/('research_'+family)
        audit = json.loads((result/'verification.json').read_text())
        comparison = json.loads((result/'comparison.json').read_text())
        assert audit['passed'] and audit['actual_seed_heads_checked'] == audit['actual_training_witnesses_checked'] == len(audit['fits']) == 48
        assert audit['frozen_register_source_heads_checked'] == len(audit['register_source_heads']) == 12
        assert audit['reused_fusion_source_heads_checked'] == len(audit['source_heads']) == 24
        assert len(audit['fixed_fusion_phases_checked']) == 16
        assert comparison['new_seed_fits'] == state['new_seed_fits'] == 48
        assert not comparison['external_test_evaluated']
        assert state['passing_development_gates'] == sum(row['development_gate_passed'] for row in comparison['candidates'])
        products = json.loads((folder/'training_products_freeze_before_audit.json').read_text())
        assert len(products['input_artifact_sha256']) == 240
        for name, expected in products['input_artifact_sha256'].items():
            check(name, expected)
        for row in audit['fits']:
            seed_folder = folder/row['arm']/row['phase']/f'seed_{row["seed"]}'
            check(seed_folder/'model.pt', row['checkpoint_sha256'])
            check(seed_folder/'training_input_witness.npz', row['training_witness_sha256'])
        for source in audit['register_source_heads']+audit['source_heads']:
            check(source['checkpoint'], source['checkpoint_sha256'])
        for row in comparison['candidates']:
            for phase in ['validation', 'fold_0', 'fold_1', 'fold_2']:
                path = folder/row['name']/phase/'predictions.npz'
                check(path, sha(path))
        for name in ['comparison.json', 'verification.json', 'freeze_integrity_after_training.json']:
            check(result/name, sha(result/name))
        assert json.loads((result/'freeze_integrity_after_training.json').read_text())['passed']
        completed[family] = {'classification_fits': 48, 'actual_task_witnesses': 48,
            'comparison_sha256': sha(result/'comparison.json'), 'verification_sha256': sha(result/'verification.json')}
    assert 'register_patch_value' in completed
    loops = {}
    for family in ['patch_loop_cuda_ready', 'gated_patch_continuation', 'readout_adaptation_continuation', 'pair_specialist_continuation']:
        folder = ROOT/'runs'/family
        formal = json.loads((folder/'protocol.json').read_text())
        check(folder/'protocol.json', sha(folder/'protocol.json'))
        for key in ['code_sha256', 'frozen_artifacts']:
            for name, expected in formal[key].items():
                check(name, expected)
        state = json.loads((folder/'state.json').read_text())
        command = Path('/proc')/str(state.get('pid'))/'cmdline'
        expected = formal.get('execution_entrypoint', 'continue_patch_loop_cuda_ready.py')
        confirmed = command.exists() and str(ROOT/expected).encode() in command.read_bytes().split(b'\0')
        if state['status'].startswith(('waiting_', 'running_')):
            assert confirmed, (family, state)
        assert state['status'] != 'failed', (family, state)
        loops[family] = {'state_at_inspection': state, 'process_confirmed_live': confirmed}
    for logical, target in targets.items():
        assert str(Path(logical).resolve(strict=True)) == target
    for path, record in physical.items():
        assert fingerprint(Path(path)) == record['fingerprint'], ('Changed before completion', path)
    dump(DEST, {'passed': True, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'base_global_audit_archive': str(base_path.relative_to(ROOT)), 'base_global_audit_sha256': sha(base_path),
        'base_completed_studies': list(base['studies']), 'completed_studies_checked_in_supplement': completed,
        'active_study_protocols_checked': active_protocols, 'continuations_checked': loops,
        'physical_files_hashed': len(physical), 'logical_records_checked': len(checked), 'sha256': checked,
        'new_classification_fits': 0, 'new_model_predictions': 0, 'external_test_evaluated': False,
        'scope_limit': 'Byte integrity and accounting supplement. The base audit is preserved; active training outputs are excluded, and task/prediction semantics rely on each completed study independent verification, not replayed here.',
        'code_sha256': protocol['code_sha256']})
    dump(OUT/'state.json', {'status': 'completed', 'pid': os.getpid(), 'passed': True,
        'completed_families_checked': list(completed), 'external_test_evaluated': False})
    print(json.dumps({'passed': True, 'completed_families': list(completed), 'physical_files': len(physical)}), flush=True)


def main():
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            run()
        except BaseException:
            dump(OUT/'state.json', {'status': 'failed', 'pid': os.getpid(), 'error': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()
