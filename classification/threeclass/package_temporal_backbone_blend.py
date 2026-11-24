"""Package only after the frozen temporal/backbone candidate passes confirmation."""
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone
from experiment import ROOT, dump
from kernel_residual_experiments import sha
from temporal_backbone_mixtures import checkpoint_path


def main():
    confirmation = ROOT/'results/research_temporal_backbone_confirmation'
    result = json.loads((confirmation/'comparison.json').read_text())
    assert result['confirmation_gate_passed'] and not result['external_test_evaluated']
    verification = json.loads((confirmation/'verification.json').read_text())
    assert verification['passed'] and verification['actual_seed_heads_checked'] == 72
    assert verification['actual_used_source_weight_states_checked'] == 90
    assert verification['actual_saved_selected_weight_states_checked'] == 108 and verification['actual_training_witnesses_checked'] == 54
    assert json.loads((confirmation/'freeze_integrity_after_confirmation.json').read_text())['passed']
    from temporal_backbone_confirmation import verify_frozen
    verify_frozen(result['protocol'])
    selection_path = ROOT/'results/research_temporal_backbone_mixtures/selection_before_repeat.json'
    selected = json.loads(selection_path.read_text())['candidate']
    assert selected['candidate_id'] == result['protocol']['selected_candidate']
    assert selected['development_gate_passed']
    weight = selected['mixture_configuration']['dropout_weight']
    assert weight == result['protocol']['selected_configuration']['weight']
    parent = json.loads((ROOT/'runs/temporal_backbone_mixtures/protocol.json').read_text())
    for name, expected in parent['source_checkpoint_sha256'].items(): assert sha(name) == expected
    folder = ROOT/'results/milestone8_temporal_backbone'
    storage = Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124/milestone8_temporal_backbone')
    storage.mkdir(parents=True, exist_ok=True)
    if not folder.exists(): folder.symlink_to(storage, target_is_directory=True)
    assert folder.resolve() == storage.resolve() and not (folder/'package.json').exists()
    hashes = {}; sources = {}
    for seed in range(3):
        for branch in ['reference', 'subspace', 'ema', 'large', 'dropout']:
            source = checkpoint_path('validation', branch, seed)
            target = folder/'frozen_package'/f'seed_{seed}'/(branch+'.pt')
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
            hashes[f'{branch}_seed{seed}'] = sha(target); sources[str(source)] = sha(source)
            assert sha(target) == sha(source)
    code = ['predict_temporal_backbone_blend.py', 'package_temporal_backbone_blend.py', 'verify_temporal_backbone_inference.py', 'audit_temporal_backbone_package.py',
        'predict_normal_large_blend_v2.py', 'experiment.py', 'normal_subspace.py', 'normal_cross_geometry.py',
        'predict_frozen_gaussian.py', 'lora_model.py', 'highres_data.py', 'extract_vit.py', 'kernel_residual_experiments.py', 'temporal_backbone_mixtures.py']
    large = json.loads((ROOT/'runs/dinov3_large/protocol.json').read_text())
    metadata = json.loads((ROOT/'features/vitb_320.json').read_text())
    package = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'candidate_id': 'milestone8_temporal_backbone_'+selected['candidate_id'],
        'manifest_sha256': sha(ROOT/'metadata/manifest.csv'), 'dropout_mixture_weight': weight,
        'class_order': ['infectious', 'noninfectious', 'normal'], 'branch_weights': {
            'reference': .1875*(1-weight)+.25*weight, 'subspace': .28125*(1-weight), 'ema': .28125*(1-weight),
            'large': .25*(1-weight), 'dropout': .75*weight},
        'base_backbone_sha256': metadata['weights_sha256'], 'large_pretrained_weights': large['pretrained_weights'],
        'model_sha256': hashes, 'source_model_sha256': sources, 'development_validation': selected['val'], 'original_patient_cv': selected['cv'],
        'selection_record': str(selection_path.relative_to(ROOT)), 'selection_record_sha256': sha(selection_path),
        'confirmation_record': str((confirmation/'comparison.json').relative_to(ROOT)), 'confirmation_record_sha256': sha(confirmation/'comparison.json'),
        'confirmation_verification': str((confirmation/'verification.json').relative_to(ROOT)), 'confirmation_verification_sha256': sha(confirmation/'verification.json'),
        'confirmation_integrity': str((confirmation/'freeze_integrity_after_confirmation.json').relative_to(ROOT)),
        'confirmation_integrity_sha256': sha(confirmation/'freeze_integrity_after_confirmation.json'),
        'code_sha256': {name: sha(ROOT/name) for name in code}, 'external_test_evaluated': False,
        'status': 'Patient-confirmed candidate pending full original-image inference verification; preserve all historical references.'}
    dump(folder/'package.json', package)
    print(json.dumps({'candidate': package['candidate_id'], 'package_sha256': sha(folder/'package.json')}), flush=True)


if __name__ == '__main__': main()
