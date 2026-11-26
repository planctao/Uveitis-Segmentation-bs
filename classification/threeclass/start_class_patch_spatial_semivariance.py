"""Construct a frozen spatial study only after completed negative predecessors and actual proofs."""
import json,os,shutil,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
from experiment import ROOT,dump
from kernel_residual_experiments import sha
from start_class_query_patch_evidence import scientific_closure
EXCLUDED={'integrity_reporting.py','refresh_recent_progress.py','report.py','write_latest_results.py'}
OUT=ROOT/'runs/class_patch_spatial_semivariance'
PREREG='results/class_patch_spatial_semivariance_preregistration.json'

def checked(record,code,artifacts):
    for section in ['code_sha256','input_artifact_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
        mapping=record.get(section,{})
        if not isinstance(mapping,dict):continue
        for name,expected in mapping.items():
            p=Path(name);p=p if p.is_absolute() else ROOT/p
            assert sha(p)==expected,name
            if section=='code_sha256':
                assert name not in EXCLUDED;code.add(name)
            else:artifacts.add(p)

def build_protocol():
    code=set();artifacts=set()
    def read(name):
        p=ROOT/name;artifacts.add(p);return json.loads(p.read_text())
    state=read('runs/class_patch_dispersion/state.json')
    assert state['status']=='completed' and state['new_seed_fits']==24 and state['passing_development_gates']==0
    observer=read('runs/class_patch_dispersion_continuation_v3/state.json')
    assert observer['status']=='completed_class_patch_dispersion_decision_requires_next_exploration'
    comparison=read('results/research_class_patch_dispersion/comparison.json')
    audit=read('results/research_class_patch_dispersion/verification.json')
    saved=read('results/research_class_patch_dispersion/independent_saved_output_review.json')
    assert audit['passed'] and audit['actual_seed_heads_checked']==audit['actual_training_witnesses_checked']==24
    assert not any(r['development_gate_passed'] for r in comparison['candidates'])
    assert saved['review_passed'] and saved['gate_passers']==0
    for name,expected in saved['input_sha256'].items():assert sha(name)==expected,name;artifacts.add(Path(name))
    parent=read('runs/class_patch_dispersion/protocol.json');checked(parent,code,artifacts)
    assert comparison['protocol']==parent
    for proof_name in ['verification_class_patch_spatial_semivariance_model.json','verification_class_patch_spatial_semivariance_confirmation_plan.json']:
        proof=read(proof_name);assert proof['passed']
        assert proof['classification_fits']==proof['classification_scores']==proof['external_images_read']==0
        checked(proof,code,artifacts)
        if 'cases' in proof:
            assert len(proof['cases'])==6 and proof['engineering_optimizer_steps']==2
            for row in proof['cases']:
                assert row['actual_images']==256 and row['ordinary_cuda_restore_bitwise']
                assert row['trainable_parameters']==row['mean_control_trainable_parameters']==4098
                assert sha(row['first_task_witness_path'])==row['first_task_witness_sha256'];artifacts.add(Path(row['first_task_witness_path']))
        else:
            assert len(proof['plans'])==len(proof['actual_original_fresh_control_engine_cases'])==6
            assert len(proof['synthetic_formula_cases'])==4 and len(proof['development_gate_cases'])==60 and len(proof['confirmation_gate_cases'])==29
            assert proof['supervised_fits']==126 and proof['actual_task_witnesses']==90
    for name in ['runs/class_patch_spatial_semivariance_preflight/protocol.json','runs/class_patch_spatial_semivariance_engine_preflight/protocol.json']:read(name)
    fresh_saved=read('results/research_class_query_patch_evidence_confirmation/independent_saved_output_review.json')
    fresh_root=read('results/class_patch_fresh_saved_review_v2_root_review.json')
    assert fresh_saved['review_passed'] and not fresh_saved['confirmation_gate_passed'] and fresh_root['passed']
    assert fresh_saved['completed_classification_fits']==126 and fresh_saved['completed_actual_task_witnesses']==90
    assert fresh_root['review_sha256']==sha(ROOT/'results/research_class_query_patch_evidence_confirmation/independent_saved_output_review.json')
    prereg=read(PREREG);assert sha(ROOT/PREREG)=='bfbdcaa415abfa0eec440a4e2e3fb6ff0c8618084f9d8052e09028e1ed79603d'
    clarification=read('results/class_patch_spatial_semivariance_graph_symmetry_clarification_v2.json')
    assert clarification['preregistration_sha256']==sha(ROOT/PREREG) and clarification['architecture_training_selection_confirmation_unchanged']
    registry=read('results/research_reference_registry.json');current=read(registry['active_reference'])
    assert sha(ROOT/registry['active_reference'])==registry['active_reference_sha256'] and current['retained_development_model']=='M8'
    means=parent['frozen_mean_controls'];assert len(means)==4
    variances=comparison['candidates'];assert len(variances)==4
    configs=[]
    for arm in prereg['configuration_plan']:
        assert arm['statistic']=='log_semivariance' and arm['pooling'] in ['uniform','source']
        configs.append({**arm,'matched_mean':'uniform_pool' if arm['pooling']=='uniform' else 'source_query_pool','matched_variance':'uniform_variance' if arm['pooling']=='uniform' else 'source_query_variance'})
    code.update(['start_class_patch_spatial_semivariance.py','verify_class_patch_spatial_semivariance_startup.py','start_class_patch_spatial_semivariance_confirmation.py','verify_class_patch_spatial_semivariance_confirmation.py','review_class_patch_spatial_semivariance_outputs.py'])
    code=scientific_closure(code-EXCLUDED);assert not (set(code)&EXCLUDED)
    return {'created_at_utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':sha(ROOT/'metadata/manifest.csv'),
        'retained_model':'M8','retained_reference':registry['active_reference'],'minimum_cv_macro_f1':current['original_cv_macro_f1'],'minimum_validation_macro_f1':current['validation_macro_f1'],
        'hypothesis':prereg['hypothesis'],'architecture':prereg['architecture_plan'],'configuration_plan':configs,'preregistered_configuration_plan':prereg['configuration_plan'],
        'control_annotations':'matched_mean and matched_variance name existing corresponding frozen controls; these metadata fields do not alter preregistered architecture.',
        'planned_new_seed_fits':24,'reused_mean_control_trajectories':24,'reused_global_variance_control_trajectories':24,'predeclared_scores':4,
        'training':prereg['fixed_training'],'fixed_fusions':prereg['fixed_fusions'],'selection':prereg['original_selection'],'confirmation_plan':prereg['confirmation_plan'],
        'frozen_mean_controls':means,'frozen_variance_controls':variances,'source_checkpoint_sha256':parent['source_checkpoint_sha256'],'source_prediction_sha256':parent['source_prediction_sha256'],
        'code_sha256':{n:sha(ROOT/n) for n in code},'frozen_artifacts':{str(p):sha(p) for p in sorted(artifacts)},
        'execution_entrypoint':'class_patch_spatial_semivariance_experiments.py','external_test_policy':'Only1717training/374validationdevelopmentimages; cached external rows NaN. No external prediction/outcome access or automatic promotion.'}

def main():
    assert not (OUT/'protocol.json').exists(),'Never duplicate a formal study.'
    proof_path=ROOT/'verification_class_patch_spatial_semivariance_startup.json';proof=json.loads(proof_path.read_text())
    assert proof['passed'] and proof['success_path_protocol_constructed_without_launch']
    for name,expected in proof['code_sha256'].items():assert sha(ROOT/name)==expected,name
    preview=ROOT/proof['protocol_preview'];assert sha(preview)==proof['protocol_preview_sha256']
    protocol=build_protocol();expected=json.loads(preview.read_text())
    assert {k:v for k,v in protocol.items() if k!='created_at_utc'}=={k:v for k,v in expected.items() if k!='created_at_utc'}
    assert shutil.disk_usage('/autodl-fs/data').free>128*1024**2
    storage=Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124/class_patch_spatial_semivariance');storage.mkdir(exist_ok=True,parents=True)
    if not OUT.exists():OUT.symlink_to(storage,target_is_directory=True)
    assert OUT.resolve()==storage.resolve()
    rule_path=ROOT/'runs/class_patch_spatial_semivariance_saved_review/protocol.json';assert not rule_path.exists()
    dump(rule_path,{'created_at_utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':protocol['manifest_sha256'],'code_sha256':sha(ROOT/'review_class_patch_spatial_semivariance_outputs.py'),'frozen_artifacts':protocol['frozen_artifacts'],'purpose':'Reconstruct saved metrics/fusions and M8+mean+variance gates after completed actual24model/task audit; no new fitting/predictions.'})
    dump(OUT/'protocol.json',protocol);dump(OUT/'state.json',{'status':'frozen_before_training','new_seed_fits':0,'external_test_evaluated':False})
    env=dict(os.environ);env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
    with (ROOT/'class_patch_spatial_semivariance.log').open('x') as log:
        process=subprocess.Popen([sys.executable,'-u',str(ROOT/protocol['execution_entrypoint'])],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    dump(OUT/'launch_record.json',{'pid':process.pid,'protocol_sha256':sha(OUT/'protocol.json'),'actual_success_preview_matched':True})
    hypothesis_path=ROOT/'results/next_class_patch_spatial_semivariance_hypothesis.json';hypothesis=json.loads(hypothesis_path.read_text())
    dump(hypothesis_path,{**hypothesis,'status':'running','training_launched':True,'pid':process.pid,'protocol':str((OUT/'protocol.json').relative_to(ROOT)),'protocol_sha256':sha(OUT/'protocol.json')})
    print(json.dumps({'pid':process.pid,'planned_new_seed_fits':24,'reused_mean_trajectories':24,'reused_variance_trajectories':24,'protocol_sha256':sha(OUT/'protocol.json')}),flush=True)
if __name__=='__main__':main()
