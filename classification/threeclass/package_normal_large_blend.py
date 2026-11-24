"""Package a confirmed normal-L/M6 mixture for development-image inference checks."""
import json
import shutil
from datetime import datetime,timezone
from pathlib import Path
from experiment import ROOT,dump
from kernel_residual_experiments import sha


def main():
    folder=ROOT/'results/milestone7';storage=Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124/milestone7')
    confirmation=ROOT/'results/research_normal_large_blend_confirmation'
    result=json.loads((confirmation/'comparison.json').read_text())
    assert result['confirmation_gate_passed'] and not result['external_test_evaluated']
    verification=json.loads((confirmation/'verification.json').read_text())
    assert verification['passed'] and verification['actual_seed_heads_checked']==54 and verification['actual_training_witnesses_checked']==18
    assert json.loads((confirmation/'freeze_integrity_after_confirmation.json').read_text())['passed']
    from normal_large_blend_confirmation import verify_frozen
    verify_frozen(result['protocol'])
    selection_record=ROOT/'results/research_normal_large_blend/selection_before_repeat.json'
    selected=json.loads(selection_record.read_text())['candidate']
    assert selected['candidate_id']==result['protocol']['selected_candidate']=='large_dual_weight0p25'
    assert selected['blend_configuration']=={'name':'large_dual_weight0p25','feature':'large_dual','large_weight':.25}
    assert selected['development_gate_passed']
    parent=json.loads((ROOT/'runs/normal_large_blend/protocol.json').read_text())
    for name,expected in parent['source_checkpoint_sha256'].items():assert sha(name)==expected
    storage.mkdir(parents=True,exist_ok=True)
    if not folder.exists():folder.symlink_to(storage,target_is_directory=True)
    assert folder.resolve()==storage.resolve() and not (folder/'package.json').exists()
    hashes={};sources={}
    for seed in range(3):
        origins={'reference':ROOT/'runs/normal_subspace/zero_control/validation',
            'subspace':ROOT/'runs/normal_subspace/normal_rank32_full/validation',
            'large':ROOT/'runs/normal_large_reference/large_dual/validation'}
        for branch,origin in origins.items():
            source=origin/f'seed_{seed}/model.pt';path=folder/'frozen_package'/f'seed_{seed}'/(branch+'.pt')
            path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,path)
            hashes[f'{branch}_seed{seed}']=sha(path);sources[str(source)]=sha(source);assert sha(path)==sha(source)
    code=['predict_normal_large_blend.py','verify_normal_large_blend_inference.py','package_normal_large_blend.py',
        'experiment.py','normal_subspace.py','predict_frozen_gaussian.py','lora_model.py','highres_data.py','extract_vit.py']
    large=json.loads((ROOT/'runs/dinov3_large/protocol.json').read_text());metadata=json.loads((ROOT/'features/vitb_320.json').read_text())
    package={'created_at_utc':datetime.now(timezone.utc).isoformat(),'candidate_id':'milestone7_normal_large_blend025',
        'manifest_sha256':sha(ROOT/'metadata/manifest.csv'),'feature':'large_dual','large_normal_weight':.25,
        'reference_weight':.1875,'normal_base_weight':.5625,'class_order':['infectious','noninfectious','normal'],
        'base_backbone_sha256':metadata['weights_sha256'],'large_pretrained_weights':large['pretrained_weights'],
        'model_sha256':hashes,'source_model_sha256':sources,'development_validation':selected['val'],'original_patient_cv':selected['cv'],
        'selection_record':str(selection_record.relative_to(ROOT)),'selection_record_sha256':sha(selection_record),
        'confirmation_record':str((confirmation/'comparison.json').relative_to(ROOT)),'confirmation_record_sha256':sha(confirmation/'comparison.json'),
        'confirmation_verification':str((confirmation/'verification.json').relative_to(ROOT)),'confirmation_verification_sha256':sha(confirmation/'verification.json'),
        'confirmation_integrity':str((confirmation/'freeze_integrity_after_confirmation.json').relative_to(ROOT)),'confirmation_integrity_sha256':sha(confirmation/'freeze_integrity_after_confirmation.json'),
        'code_sha256':{name:sha(ROOT/name) for name in code},'external_test_evaluated':False,
        'status':'Development-confirmed candidate pending complete original-image inference verification; preserve M6 frozen reference. No automatic external testing.'}
    dump(folder/'package.json',package)
    print({'candidate':package['candidate_id'],'package_sha256':sha(folder/'package.json')},flush=True)


if __name__=='__main__':main()
