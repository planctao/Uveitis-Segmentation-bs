"""Freeze and launch the formal normal-L residual development experiment."""
import json,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
from experiment import ROOT,dump
from kernel_residual_experiments import sha


def main():
    out=ROOT/'runs/normal_large_residual';storage=Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124/normal_large_residual');storage.mkdir(parents=True,exist_ok=True)
    if not out.exists():out.symlink_to(storage,target_is_directory=True)
    assert out.resolve()==storage.resolve() and not (out/'protocol.json').exists()
    current=json.loads((ROOT/'results/current_research_reference.json').read_text());assert current['retained_development_model']=='M7'
    preflight=json.loads((ROOT/'verification_normal_large_residual_draft.json').read_text());assert preflight['passed']
    for seed in [20280721,20280731]:
        for protocol in ROOT.glob('runs/*/protocol.json'):assert str(seed) not in protocol.read_text()
    reference=ROOT/'results/milestone5/selection_before_test.json';m6=ROOT/'results/milestone6/selection_before_test.json';m7=ROOT/'results/current_research_reference.json'
    frozen=[reference,m6,m7,ROOT/'results/milestone7_normal_large_blend/package.json',ROOT/'results/milestone7_normal_large_blend/selection_development_only.json',ROOT/'features/multiscale_vit.npz',ROOT/'features/multiscale_vit.json',ROOT/'verification_normal_large_residual_draft.json']
    large=json.loads((ROOT/'runs/dinov3_large/protocol.json').read_text());frozen += [Path(large['feature_storage']),Path(large['feature_storage']).with_suffix('.json'),ROOT/'runs/dinov3_large/protocol.json']
    protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':sha(ROOT/'metadata/manifest.csv'),'reference':'results/milestone5/selection_before_test.json','current_reference':'results/current_research_reference.json','outer_split_seeds':[20280721,20280731],
        'configuration_plan':[{'name':f'{scope}_rank{rank}','scope':scope,'rank':rank} for rank in [16,32] for scope in ['global','all']],
        'minimum_cv_macro_f1':current['original_cv_macro_f1'],'minimum_validation_macro_f1':current['validation_macro_f1'],'planned_new_seed_fits':48,'external_test_policy':'Development rows only; no external outcomes or predictions loaded.',
        'hypothesis':'Identity-initialized low-rank normal-L residual can complement M7 while preserving the normal-B pathway at initialization. Four rank/scope configurations, each 12 heads, plus four fixed .75 residual/.25 M5 scores.',
        'fresh_confirmation_gate':'Highest qualifying original CV candidate must then exceed M7, M6, M5 and fixed Gaussian on two unused patient three-fold repeats by mean Macro-F1 and paired patient bootstrap lower endpoint.',
        'frozen_artifacts':{str(p):sha(p) for p in frozen},'code_sha256':{name:sha(ROOT/name) for name in ['normal_large_residual.py','normal_large_residual_experiments.py','start_normal_large_residual.py','experiment.py','normal_subspace.py','dinov3_large_data.py','kernel_residual_experiments.py','milestone2.py','verify_normal_large_residual.py']}}
    dump(out/'protocol.json',protocol);dump(out/'state.json',{'status':'frozen_before_training','new_seed_fits':0,'external_test_evaluated':False})
    env=dict(os.environ);env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
    with (ROOT/'normal_large_residual.log').open('a') as log:process=subprocess.Popen([sys.executable,'-u',str(ROOT/'normal_large_residual_experiments.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    dump(ROOT/'results/next_normal_large_residual_hypothesis.json',{**preflight,'status':'formal_running','protocol':'runs/normal_large_residual/protocol.json','protocol_sha256':sha(out/'protocol.json'),'pid':process.pid,'formal_protocol_created':True})
    print(json.dumps({'pid':process.pid,'protocol_sha256':sha(out/'protocol.json'),'planned_new_seed_fits':48}),flush=True)


if __name__=='__main__':main()
