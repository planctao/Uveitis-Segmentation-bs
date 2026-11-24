"""Recover CLI verification with matching image batch size; preserve frozen code."""
import json
import subprocess
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
from experiment import ROOT, dump
from kernel_residual_experiments import sha
from predict_register_blend import FrozenRegisterBlendPredictor


def main():
    torch.set_num_threads(4); folder = ROOT/'results/milestone8_register_blend'
    verification = json.loads((folder/'inference_verification.json').read_text())
    assert verification['passed'] and verification['validation_images_checked'] == 374
    assert verification['real_cached_and_original_images_checked'] == 6
    selection = json.loads((folder/'selection_development_only.json').read_text())
    assert sha(folder/'package.json') == selection['package_sha256']
    assert sha(folder/'inference_verification.json') == selection['inference_verification_sha256']
    recovery = {'time_utc': datetime.now(timezone.utc).isoformat(),
        'reason': 'Original CLI assertion compared a single-image fresh DINO FP16 forward with the first element of a three-image fresh forward. All374head and sixsame-batch original/cache checks already passed. Compare CLI to equal-batch fresh path; separately report batching error and require unchanged decisions/established live-cache mean probability tolerance.',
        'new_seed_fits': 0, 'classification_scores_evaluated': 0,
        'code_sha256': {'verify_register_cli_v2.py': sha(ROOT/'verify_register_cli_v2.py')},
        'input_artifact_sha256': {str(folder/name): sha(folder/name) for name in ['package.json', 'inference_verification.json', 'selection_development_only.json']},
        'external_test_evaluated': False}
    assert not (folder/'cli_recovery_protocol.json').exists(); dump(folder/'cli_recovery_protocol.json', recovery)
    frame = pd.read_csv(ROOT/'metadata/manifest.csv'); ids = verification['real_validation_indices']
    paths = frame.iloc[ids].path.tolist(); predictor = FrozenRegisterBlendPredictor()
    batched = predictor.predict_images(paths, batch=3); per_image = predictor.predict_images(paths, batch=1)
    assert np.array_equal(batched.argmax(1), per_image.argmax(1))
    assert np.abs(batched-per_image).mean() < .001
    process = subprocess.run([sys.executable, str(ROOT/'predict_register_blend.py'), '--image', paths[0]], cwd=ROOT,
        check=True, text=True, capture_output=True)
    cli = json.loads(process.stdout); row = cli['predictions'][0]
    probability = np.array([row['probabilities'][name] for name in ['感染', '非感染', '正常']])
    assert np.allclose(probability, per_image[0], atol=6e-5, rtol=6e-5)
    assert probability.argmax() == per_image[0].argmax()
    for path, expected in recovery['input_artifact_sha256'].items(): assert sha(path) == expected
    dump(folder/'cli_verification.json', {'passed': True, 'image': paths[0], 'candidate_id': cli['candidate_id'],
        'maximum_probability_error': float(np.abs(probability-per_image[0]).max()), 'exit_code': process.returncode,
        'six_image_batch1_vs_batch3_maximum_error': float(np.abs(batched-per_image).max()),
        'six_image_batch1_vs_batch3_mean_error': float(np.abs(batched-per_image).mean()),
        'all_six_batching_decisions_identical': True,
        'comparison': 'CLI oneimage vs same-batch fresh API; original/cache same-batch checks preserved. FP16 batching numerical variation reported separately.',
        'recovery_protocol_sha256': sha(folder/'cli_recovery_protocol.json'), 'recovery_code_sha256': sha(ROOT/'verify_register_cli_v2.py'),
        'external_images_or_outcomes_used': False})
    dump(folder/'selection_development_only.json', {**selection, 'cli_verification_sha256': sha(folder/'cli_verification.json'),
        'cli_recovery_protocol_sha256': sha(folder/'cli_recovery_protocol.json')})
    print(json.dumps(json.loads((folder/'cli_verification.json').read_text())), flush=True)


if __name__ == '__main__': main()
