"""Copy exactly the nine original-development source models after confirmation."""
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone
from experiment import ROOT, dump
from kernel_residual_experiments import sha


def checkpoint_path(branch, seed):
    source = {'reference': 'runs/normal_subspace/zero_control', 'subspace': 'runs/normal_subspace/normal_rank32_full',
        'token': 'runs/large_token_evidence/registers'}[branch]
    return ROOT/source/'validation'/f'seed_{seed}/model.pt'


def main():
    confirmation = ROOT/'results/research_large_token_confirmation'
    result = json.loads((confirmation/'comparison.json').read_text()); verification = json.loads((confirmation/'verification.json').read_text())
    assert result['confirmation_gate_passed'] and not result['external_test_evaluated']
    assert verification['passed'] and verification['actual_seed_heads_checked'] == 72 and verification['actual_training_witnesses_checked'] == 36
    assert json.loads((confirmation/'freeze_integrity_after_confirmation.json').read_text())['passed']
    from large_token_confirmation import verify_frozen
    verify_frozen(result['protocol'])
    selection_path = ROOT/'results/research_large_token_evidence/selection_before_repeat.json'
    selected = json.loads(selection_path.read_text())['candidate']
    assert selected['candidate_id'] == result['protocol']['selected_candidate'] == 'registers_m7replace'
    assert selected['development_gate_passed']
    folder = ROOT/'results/milestone8_register_blend'
    storage = Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124/milestone8_register_blend')
    storage.mkdir(parents=True, exist_ok=True)
    if not folder.exists(): folder.symlink_to(storage, target_is_directory=True)
    assert folder.resolve() == storage.resolve() and not (folder/'package.json').exists()
    hashes, sources = {}, {}
    for seed in range(3):
        for branch in ['reference', 'subspace', 'token']:
            source = checkpoint_path(branch, seed); target = folder/'frozen_package'/f'seed_{seed}'/(branch+'.pt')
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
            assert sha(target) == sha(source); hashes[f'{branch}_seed{seed}'] = sha(target); sources[str(source)] = sha(source)
    code = ['predict_register_blend.py', 'package_register_blend.py', 'verify_register_inference.py', 'audit_register_package.py',
        'predict_normal_large_blend_v2.py', 'large_token_evidence.py', 'audit_large_token_evidence.py',
        'experiment.py', 'normal_subspace.py', 'predict_frozen_gaussian.py', 'lora_model.py', 'highres_data.py', 'extract_vit.py',
        'kernel_residual_experiments.py', 'audit_normal_tangent.py']
    large = json.loads((ROOT/'runs/dinov3_large/protocol.json').read_text()); metadata = json.loads((ROOT/'features/vitb_320.json').read_text())
    package = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'candidate_id': 'milestone8_register_blend',
        'manifest_sha256': sha(ROOT/'metadata/manifest.csv'), 'class_order': ['infectious', 'noninfectious', 'normal'],
        'branch_weights': {'reference': .1875, 'subspace': .5625, 'token': .25}, 'token_stream': 'registers',
        'base_backbone_sha256': metadata['weights_sha256'], 'large_pretrained_weights': large['pretrained_weights'],
        'model_sha256': hashes, 'source_model_sha256': sources, 'development_validation': selected['val'], 'original_patient_cv': selected['cv'],
        'selection_record': str(selection_path.relative_to(ROOT)), 'selection_record_sha256': sha(selection_path),
        'confirmation_record': str((confirmation/'comparison.json').relative_to(ROOT)), 'confirmation_record_sha256': sha(confirmation/'comparison.json'),
        'confirmation_verification': str((confirmation/'verification.json').relative_to(ROOT)), 'confirmation_verification_sha256': sha(confirmation/'verification.json'),
        'confirmation_integrity': str((confirmation/'freeze_integrity_after_confirmation.json').relative_to(ROOT)), 'confirmation_integrity_sha256': sha(confirmation/'freeze_integrity_after_confirmation.json'),
        'code_sha256': {name: sha(ROOT/name) for name in code}, 'external_test_evaluated': False,
        'status': 'Patient-confirmed candidate pending original-image inference and independent package verification.'}
    dump(folder/'package.json', package)
    print(json.dumps({'candidate': package['candidate_id'], 'package_sha256': sha(folder/'package.json')}), flush=True)


if __name__ == '__main__': main()
