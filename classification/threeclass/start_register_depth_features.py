"""Freeze development image bytes, source banks and the depth extraction path."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
import pandas as pd
from experiment import ROOT, dump
from kernel_residual_experiments import sha
from register_depth_features import TOKEN_KEYS


def main():
    proof_path = ROOT/'verification_register_depth_preflight.json'; proof = json.loads(proof_path.read_text())
    assert proof['passed'] and len(proof['actual_images_recomputed']) == 6
    for name, expected in proof['code_sha256'].items(): assert sha(ROOT/name) == expected
    hypothesis = ROOT/'results/next_register_depth_hypothesis.json'; draft = json.loads(hypothesis.read_text())
    assert draft['scores_evaluated'] == 0
    old_proof_path = ROOT/'verification_large_token_feature_bank.json'; old_proof = json.loads(old_proof_path.read_text())
    assert old_proof['passed'] and old_proof['development_images_checked'] == 2091
    assert sha(old_proof['feature_storage']) == old_proof['feature_bank_sha256']
    storage = Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124/register_depth_features')
    storage.mkdir(parents=True, exist_ok=True); out = ROOT/'runs/register_depth_features'
    if not out.exists(): out.symlink_to(storage, target_is_directory=True)
    assert out.resolve() == storage.resolve() and not (out/'protocol.json').exists()
    frame = pd.read_csv(ROOT/'metadata/manifest.csv'); development = frame.loc[frame.split.isin(['train', 'val'])]
    assert len(development) == 2091 and development.index.tolist() == list(range(2091))
    source = json.loads((ROOT/'runs/dinov3_large/protocol.json').read_text())
    cache_hashes = {path: sha(path) for path in development.cache_path}
    for row in proof['actual_images_recomputed']:
        assert cache_hashes[frame.iloc[row['development_index']].cache_path] == row['cache_sha256']
    files = [ROOT/'metadata/manifest.csv', proof_path, old_proof_path, ROOT/'runs/dinov3_large/protocol.json',
        Path(source['feature_storage']), Path(source['feature_storage']).with_suffix('.json'), ROOT/'verification_dinov3_large.json',
        Path(old_proof['feature_storage']), Path(old_proof['feature_storage']).with_suffix('.json'),
        ROOT/'runs/large_token_features/protocol.json']
    code = ['register_depth_features.py', 'verify_register_depth_preflight.py', 'extract_register_depth_features.py',
        'verify_register_depth_bank.py', 'start_register_depth_features.py', 'extract_dinov3_large.py', 'experiment.py', 'kernel_residual_experiments.py']
    target = storage/'features/dinov3_register_depth_development.npz'; assert not target.exists()
    protocol = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'manifest_sha256': sha(ROOT/'metadata/manifest.csv'),
        'pretrained_weights': source['pretrained_weights'], 'feature_storage': str(target), 'hypothesis': draft['hypothesis'],
        'feature_plan': draft['feature_plan'], 'token_counts': {key: 4 for key in TOKEN_KEYS}, 'scales': [320, 448],
        'token_dtypes': {key: 'float16' if key == 'final_registers' else 'float32' for key in TOKEN_KEYS},
        'depths': ['raw_block7', 'raw_block15', 'final_norm'],
        'final_register_control_source': 'Check freshforward versus originalbank on everyimage; store exact historical finalregisterbytes.',
        'development_indices': list(range(2091)), 'cache_image_sha256': cache_hashes,
        'classification_fits': 0, 'classification_scores': 0, 'external_images_read': 0,
        'frozen_artifacts': {str(path): sha(path) for path in files}, 'code_sha256': {name: sha(ROOT/name) for name in code}}
    dump(out/'protocol.json', protocol); dump(out/'state.json', {'status': 'frozen_before_extraction', 'completed_images': 0, 'classification_scores': 0})
    env = dict(os.environ); env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    with (ROOT/'register_depth_features.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT/'extract_register_depth_features.py')], cwd=ROOT,
            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    dump(hypothesis, {**draft, 'status': 'extracting_development_features', 'feature_protocol': 'runs/register_depth_features/protocol.json',
        'feature_protocol_sha256': sha(out/'protocol.json'), 'feature_extraction_pid': process.pid})
    print(json.dumps({'pid': process.pid, 'protocol_sha256': sha(out/'protocol.json'), 'development_images': 2091,
        'classification_fits': 0, 'classification_scores': 0}), flush=True)


if __name__ == '__main__': main()
