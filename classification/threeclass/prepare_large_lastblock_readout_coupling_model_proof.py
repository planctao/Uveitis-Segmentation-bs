"""Freeze the preregistered joint-readout model engineering before execution."""
import hashlib
import inspect
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from experiment import ROOT, dump
from start_class_query_patch_evidence import scientific_closure
from large_lastblock_readout_coupling import SEED_BASE

EXCLUDED = {'integrity_reporting.py', 'refresh_recent_progress.py', 'report.py', 'write_latest_results.py'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    out = ROOT/'runs/large_lastblock_readout_coupling_model_preflight'
    assert not (out/'protocol.json').exists(), 'Never duplicate engineering.'
    prereg_path = ROOT/'results/large_lastblock_readout_coupling_preregistration.json'
    prereg = json.loads(prereg_path.read_text())
    previous_path = ROOT/'verification_large_lastblock_delta_lora_model.json'
    previous = json.loads(previous_path.read_text())
    assert previous['passed'] and previous['classification_fits'] == previous['classification_scores'] == 0
    root_path = ROOT/'results/large_lastblock_delta_lora_completed_root_review.json'
    reviewed = json.loads(root_path.read_text())
    assert reviewed['passed'] and reviewed['gate_passers'] == 0 and not reviewed['confirmation_launched']
    assert reviewed['control_never_nominated']
    for name, expected in prereg['predecessor_completed_record_sha256'].items():
        assert sha(ROOT/name) == expected, name
    for name, expected in previous['code_sha256'].items():
        assert sha(ROOT/name) == expected, name
    code = set(previous['code_sha256'])-EXCLUDED
    code.update(['large_lastblock_readout_coupling.py', 'verify_large_lastblock_readout_coupling_model.py',
                 'prepare_large_lastblock_readout_coupling_model_proof.py'])
    code = scientific_closure(code-EXCLUDED)
    assert not set(code).intersection(EXCLUDED)
    inputs = dict(previous['input_artifact_sha256'])
    inputs.update({str(path): sha(path) for path in [previous_path, prereg_path, root_path]})
    for name in prereg['predecessor_completed_record_sha256']:
        path = ROOT/name
        inputs[str(path)] = sha(path)
    frame = pd.read_csv(ROOT/'metadata/manifest.csv')
    fit = np.flatnonzero(frame.split.eq('train'))
    ids = fit[np.random.default_rng(SEED_BASE).permutation(len(fit))[:256]]
    assert len(ids) == 256
    actual_images = {}
    for name in frame.iloc[ids].cache_path:
        path = Path(name)
        actual_images[str(path)] = sha(path)
    inputs.update(actual_images)
    packages = dict(previous['package_implementation_sha256'])
    for obj in [np, pd]:
        path = Path(inspect.getfile(obj)).resolve()
        packages[str(path)] = sha(path)
    for name, expected in inputs.items():
        assert sha(name) == expected, name
    protocol = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'execution_entrypoint': 'verify_large_lastblock_readout_coupling_model.py',
                'preregistration': str(prereg_path.relative_to(ROOT)), 'preregistration_sha256': sha(prereg_path),
                'indices': ids.tolist(), 'actual_training_image_sha256': actual_images,
                'code_sha256': {name: sha(ROOT/name) for name in code},
                'input_artifact_sha256': inputs, 'package_implementation_sha256': packages,
                'excluded_reporting_modules_before_closure': sorted(EXCLUDED),
                'actual_model_cases': 3, 'actual_images_per_case': 256, 'engineering_optimizer_steps': 3,
                'classification_fits': 0, 'classification_scores': 0,
                'scope': 'Three actual256 complete tasks/all gradients, zero source anchors, updated gradients, frozen states, actual8 checkpoint/direct and ordinary CUDA restore. No candidate accuracy, classifier fit or formal launch.'}
    dump(out/'protocol.json', protocol)
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    with (out/'proof.log').open('x') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT/protocol['execution_entrypoint'])],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    dump(out/'launch_record.json', {'pid': process.pid, 'protocol_sha256': sha(out/'protocol.json')})
    hypothesis = ROOT/'results/next_large_lastblock_readout_coupling_hypothesis.json'
    data = json.loads(hypothesis.read_text())
    dump(hypothesis, {**data, 'status': 'actual256_model_engineering_running', 'model_proof_pid': process.pid,
                      'model_proof_protocol': str((out/'protocol.json').relative_to(ROOT))})
    print(json.dumps({'pid': process.pid, 'scientific_modules': len(code), 'inputs': len(inputs),
                      'actual_training_images': len(actual_images), 'protocol_sha256': sha(out/'protocol.json'),
                      'classification_fits': 0}), flush=True)


if __name__ == '__main__':
    main()
