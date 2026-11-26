"""Freeze the new shared engine, patient plans and decision tests before execution."""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from experiment import ROOT, dump
from start_class_query_patch_evidence import scientific_closure
from large_lastblock_delta_lora_engine import sha

EXCLUDED = {'integrity_reporting.py', 'refresh_recent_progress.py', 'report.py', 'write_latest_results.py'}


def main():
    out = ROOT/'runs/large_lastblock_readout_coupling_engine_preflight'
    assert not (out/'protocol.json').exists(), 'Never duplicate engineering.'
    previous_path = ROOT/'verification_large_lastblock_readout_coupling_model.json'
    previous = json.loads(previous_path.read_text())
    assert previous['passed'] and len(previous['cases']) == 3 and previous['engineering_optimizer_steps'] == 3
    assert previous['classification_fits'] == previous['classification_scores'] == 0
    for name, expected in previous['code_sha256'].items():
        assert sha(ROOT/name) == expected, name
    code = set(previous['code_sha256'])-EXCLUDED
    code.update(['large_lastblock_readout_coupling_engine.py', 'large_lastblock_readout_coupling_protocol.py',
                 'verify_large_lastblock_readout_coupling_engine.py',
                 'prepare_large_lastblock_readout_coupling_engine_proof.py'])
    code = scientific_closure(code-EXCLUDED)
    assert not set(code).intersection(EXCLUDED)
    inputs = dict(previous['input_artifact_sha256'])
    inputs[str(previous_path)] = sha(previous_path)
    for case in previous['cases']:
        path = case['actual_first_task_witness']
        assert sha(path) == case['actual_first_task_witness_sha256']
        inputs[path] = sha(path)
    parent = ROOT/'runs/large_lastblock_readout_coupling_model_preflight/protocol.json'
    inputs[str(parent)] = sha(parent)
    protocol = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'execution_entrypoint': 'verify_large_lastblock_readout_coupling_engine.py',
                'code_sha256': {name: sha(ROOT/name) for name in code}, 'input_artifact_sha256': inputs,
                'package_implementation_sha256': previous['package_implementation_sha256'],
                'excluded_reporting_modules_before_closure': sorted(EXCLUDED),
                'actual_shared_engine_cases': 3, 'actual_images_per_case': 256,
                'classification_fits': 0, 'classification_scores': 0,
                'scope': 'Actual common engine tasks, warmstarts, predictions and restoration; patient plans and decision boundaries. Worker wiring and formal no-launch startup remain separate requirements.'}
    dump(out/'protocol.json', protocol)
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    with (out/'proof.log').open('x') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT/protocol['execution_entrypoint'])],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    dump(out/'launch_record.json', {'pid': process.pid, 'protocol_sha256': sha(out/'protocol.json')})
    hypothesis = ROOT/'results/next_large_lastblock_readout_coupling_hypothesis.json'
    data = json.loads(hypothesis.read_text())
    dump(hypothesis, {**data, 'status': 'actual_shared_engine_patient_gate_engineering_running',
                      'model_proof_sha256': sha(previous_path), 'engine_proof_pid': process.pid,
                      'engine_proof_protocol': str((out/'protocol.json').relative_to(ROOT))})
    print(json.dumps({'pid': process.pid, 'scientific_modules': len(code), 'inputs': len(inputs),
                      'protocol_sha256': sha(out/'protocol.json'), 'classification_fits': 0}), flush=True)


if __name__ == '__main__':
    main()
