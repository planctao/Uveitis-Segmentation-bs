"""Freeze source images and extraction before producing any additional token features."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
import pandas as pd
from experiment import ROOT, dump
from kernel_residual_experiments import sha
from base_token_features import TOKEN_KEYS, source_protocol


def main():
    proof_path = ROOT/'verification_base_token_feature_preflight.json'; proof = json.loads(proof_path.read_text())
    assert proof['passed'] and len(proof['actual_images_recomputed']) == 6
    for name, expected in proof['code_sha256'].items(): assert sha(ROOT/name) == expected
    hypothesis = ROOT/'results/next_base_token_evidence_hypothesis.json'; draft = json.loads(hypothesis.read_text())
    assert draft['scores_evaluated'] == 0
    storage = Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124/base_token_features'); storage.mkdir(parents=True, exist_ok=True)
    out = ROOT/'runs/base_token_features'
    if not out.exists(): out.symlink_to(storage, target_is_directory=True)
    assert out.resolve() == storage.resolve() and not (out/'protocol.json').exists()
    frame = pd.read_csv(ROOT/'metadata/manifest.csv'); development = frame.loc[frame.split.isin(['train', 'val'])]
    assert len(development) == 2091 and development.index.tolist() == list(range(2091))
    source = source_protocol()
    cache_hashes = {path: sha(path) for path in development.cache_path}
    for row in proof['actual_images_recomputed']:
        assert cache_hashes[frame.iloc[row['development_index']].cache_path] == row['cache_sha256']
    files = [ROOT/'metadata/manifest.csv', proof_path, ROOT/'features/vitb_320.json', ROOT/'features/vitb_448.json',
        Path(source['feature_storage']), Path(source['feature_storage']).with_suffix('.json'), Path(source['pretrained_weights']['storage'])]
    code = ['base_token_features.py', 'verify_base_token_feature_preflight.py', 'extract_base_token_features.py',
        'verify_base_token_feature_bank.py', 'start_base_token_features.py', 'extract_dinov3_large.py', 'extract_vit.py',
        'experiment.py', 'kernel_residual_experiments.py']
    target = storage/'features/dinov3_base_token_development.npz'; assert not target.exists()
    protocol = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'manifest_sha256': sha(ROOT/'metadata/manifest.csv'),
        'pretrained_weights': source['pretrained_weights'], 'feature_storage': str(target), 'hypothesis': draft['hypothesis'],
        'feature_plan': draft['feature_plan'], 'token_counts': dict(zip(TOKEN_KEYS, [9, 9, 4, 4])), 'scales': [320, 448],
        'late_region_control_source':'Freshforward checked against sourcebank on everyimage; save originalfrozen region control exactly to avoid numerical drift.',
        'development_indices': list(range(2091)), 'cache_image_sha256': cache_hashes,
        'classification_fits': 0, 'classification_scores': 0, 'external_images_read': 0,
        'frozen_artifacts': {str(path): sha(path) for path in files}, 'code_sha256': {name: sha(ROOT/name) for name in code}}
    dump(out/'protocol.json', protocol); dump(out/'state.json', {'status': 'frozen_before_extraction', 'completed_images': 0, 'classification_scores': 0})
    env = dict(os.environ); env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    with (ROOT/'base_token_features.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT/'extract_base_token_features.py')], cwd=ROOT,
            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    dump(hypothesis, {**draft, 'status': 'extracting_development_features', 'feature_protocol': 'runs/base_token_features/protocol.json',
        'feature_protocol_sha256': sha(out/'protocol.json'), 'feature_extraction_pid': process.pid, 'scores_evaluated': 0})
    print(json.dumps({'pid': process.pid, 'protocol_sha256': sha(out/'protocol.json'), 'development_images': 2091,
        'classification_fits': 0, 'classification_scores': 0}), flush=True)


if __name__ == '__main__': main()
