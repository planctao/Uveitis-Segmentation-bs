"""Freeze a development-confirmed B/L candidate for inference verification only."""
import json
import shutil
from datetime import datetime,timezone
from pathlib import Path
from experiment import ROOT,dump
from kernel_residual_experiments import sha


def main():
    folder=ROOT/'results/milestone7';storage=Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124/milestone7')
    confirmation=ROOT/'results/research_normal_large_reference_confirmation';result=json.loads((confirmation/'comparison.json').read_text())
    assert result['confirmation_gate_passed'] and not result['external_test_evaluated']
    assert json.loads((confirmation/'verification.json').read_text())['passed'] and json.loads((confirmation/'freeze_integrity_after_confirmation.json').read_text())['passed']
    selected=json.loads((ROOT/'results/research_normal_large_reference/selection_before_repeat.json').read_text())['candidate'];assert selected['candidate_id']==result['protocol']['selected_candidate']=='base_dual_large_dual_fusion075'
    assert selected['subspace_weight']==.75 and selected['development_gate_passed']
    storage.mkdir(parents=True,exist_ok=True)
    if not folder.exists():folder.symlink_to(storage,target_is_directory=True)
    assert folder.resolve()==storage.resolve() and not (folder/'package.json').exists()
    hashes={};sources={}
    for seed in range(3):
        for branch,source in [('reference',ROOT/'runs/normal_subspace/zero_control/validation'/f'seed_{seed}/model.pt'),('large',ROOT/'runs/normal_large_reference/base_dual_large_dual/validation'/f'seed_{seed}/model.pt')]:
            path=folder/'frozen_package'/f'seed_{seed}'/(branch+'.pt');path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,path)
            hashes[f'{branch}_seed{seed}']=sha(path);sources[str(source)]=sha(source);assert sha(path)==sha(source)
    code=['predict_large_normal_reference.py','verify_large_normal_inference.py','package_large_normal_reference.py','experiment.py','normal_subspace.py','predict_frozen_gaussian.py','lora_model.py','highres_data.py','extract_vit.py']
    for name in code:assert (ROOT/name).is_file()
    large=json.loads((ROOT/'runs/dinov3_large/protocol.json').read_text())
    metadata=json.loads((ROOT/'features/vitb_320.json').read_text())
    package={'created_at_utc':datetime.now(timezone.utc).isoformat(),'candidate_id':'milestone7_large_normal_reference_fusion075','manifest_sha256':sha(ROOT/'metadata/manifest.csv'),
        'feature':'base_dual_large_dual','large_normal_weight':.75,'class_order':['infectious','noninfectious','normal'],'base_backbone_sha256':metadata['weights_sha256'],
        'large_pretrained_weights':large['pretrained_weights'],'model_sha256':hashes,'source_model_sha256':sources,'development_validation':selected['val'],'original_patient_cv':selected['cv'],
        'selection_record':'results/research_normal_large_reference/selection_before_repeat.json','selection_record_sha256':sha(ROOT/'results/research_normal_large_reference/selection_before_repeat.json'),
        'confirmation_record':'results/research_normal_large_reference_confirmation/comparison.json','confirmation_record_sha256':sha(confirmation/'comparison.json'),
        'confirmation_verification':'results/research_normal_large_reference_confirmation/verification.json','confirmation_verification_sha256':sha(confirmation/'verification.json'),
        'confirmation_integrity':'results/research_normal_large_reference_confirmation/freeze_integrity_after_confirmation.json','confirmation_integrity_sha256':sha(confirmation/'freeze_integrity_after_confirmation.json'),
        'code_sha256':{name:sha(ROOT/name) for name in code},'external_test_evaluated':False,
        'status':'Development-confirmed candidate, pending complete original-image inference verification; M6 external reference preserved. No automatic external testing.'}
    dump(folder/'package.json',package)
    print({'candidate':package['candidate_id'],'package_sha256':sha(folder/'package.json')},flush=True)


if __name__=='__main__':main()
