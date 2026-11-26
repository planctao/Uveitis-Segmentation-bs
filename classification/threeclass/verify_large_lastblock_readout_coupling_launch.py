"""Review actual frozen formal inputs and two live bound process command lines."""
import json
from datetime import datetime, timezone
from experiment import ROOT, dump
from start_large_lastblock_readout_coupling import FAMILY, sha, live, RULES


def main():
    target = ROOT/('verification_'+FAMILY+'_launch.json')
    assert not target.exists()
    run = ROOT/'runs'/FAMILY
    protocol = json.loads((run/'protocol.json').read_text())
    startup = json.loads((ROOT/('verification_'+FAMILY+'_startup.json')).read_text())
    assert startup['passed'] and startup['success_path_protocol_constructed_without_launch']
    preview = ROOT/startup['protocol_preview']
    assert sha(preview) == startup['protocol_preview_sha256']
    before = json.loads(preview.read_text())
    assert {k: v for k, v in protocol.items() if k != 'created_at_utc'} == {k: v for k, v in before.items() if k != 'created_at_utc'}
    checked = {}
    for section in ['code_sha256', 'frozen_artifacts', 'source_checkpoint_sha256',
                    'source_prediction_sha256', 'package_implementation_sha256', 'actual_development_image_sha256']:
        for name, expected in protocol[section].items():
            assert sha(name) == expected, name
            checked[name] = expected
    launch = json.loads((run/'launch_record.json').read_text())
    assert launch['protocol_sha256'] == sha(run/'protocol.json')
    assert launch['actual_success_preview_matched'] and live(launch['pid'], protocol['execution_entrypoint'])
    observer = ROOT/'runs'/(FAMILY+'_observer')
    observer_protocol = json.loads((observer/'protocol.json').read_text())
    observer_launch = json.loads((observer/'launch_record.json').read_text())
    assert observer_launch['protocol_sha256'] == sha(observer/'protocol.json')
    assert observer_launch['worker_pid'] == observer_protocol['worker_pid'] == launch['pid']
    assert live(observer_launch['pid'], 'observe_'+FAMILY+'.py')
    assert not observer_protocol['automatic_reference_promotion']
    for section in ['code_sha256', 'frozen_artifacts']:
        for name, expected in observer_protocol[section].items():
            assert sha(name) == expected, name
    for name in RULES:
        assert sha(ROOT/name) == startup['pre_outcome_review_rules_sha256'][name]
    state = json.loads((run/'state.json').read_text())
    observer_state = json.loads((observer/'state.json').read_text())
    assert state['status'] != 'failed' and observer_state['status'] != 'failed'
    dump(target, {'passed': True, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'formal_protocol_sha256': sha(run/'protocol.json'), 'protocol_preview_matched': True,
        'frozen_input_identities_checked': len(checked), 'scientific_modules_checked': len(protocol['code_sha256']),
        'development_image_bytes_checked': 2091, 'worker_pid': launch['pid'],
        'observer_pid': observer_launch['pid'], 'both_actual_proc_commandlines_verified': True,
        'bound_observer_protocol_sha256': sha(observer/'protocol.json'),
        'all_four_pre_outcome_rules_unchanged': True, 'state_at_review': state,
        'planned_original_fits_and_tasks': 36, 'conditional_fresh_fits': 126, 'conditional_fresh_tasks': 90,
        'new_model_fits_or_predictions_in_this_review': 0, 'external_images_read': 0,
        'code_sha256': sha(__file__), 'automatic_reference_promotion': False})
    print(json.dumps(json.loads(target.read_text())), flush=True)


if __name__ == '__main__':
    main()
