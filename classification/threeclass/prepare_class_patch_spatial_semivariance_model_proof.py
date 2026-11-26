"""Freeze the spatial model proof before actual engineering evaluation."""
import json,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
from experiment import ROOT,dump
from kernel_residual_experiments import sha
from start_class_query_patch_evidence import scientific_closure
EXCLUDED={'integrity_reporting.py','refresh_recent_progress.py','report.py','write_latest_results.py'}
def main():
    out=ROOT/'runs/class_patch_spatial_semivariance_preflight'
    assert not (out/'protocol.json').exists()
    parent=json.loads((ROOT/'runs/class_patch_dispersion_preflight/protocol.json').read_text())
    code=set(parent['code_sha256'])-EXCLUDED
    code.update(['class_patch_spatial_semivariance.py','audit_class_patch_spatial_semivariance.py','class_patch_spatial_semivariance_witness.py','verify_class_patch_spatial_semivariance_model.py','prepare_class_patch_spatial_semivariance_model_proof.py'])
    code=scientific_closure(code-EXCLUDED);assert not (set(code)&EXCLUDED)
    for name,expected in parent['code_sha256'].items():
        if name not in EXCLUDED:assert sha(ROOT/name)==expected,name
    artifacts=dict(parent['input_artifact_sha256'])
    names=['runs/class_patch_dispersion_preflight/protocol.json','results/dispersion_scientific_code_inventory_recovery_v2.json','results/class_patch_spatial_semivariance_preregistration.json','results/class_patch_spatial_semivariance_graph_symmetry_clarification_v2.json','results/research_class_query_patch_evidence_confirmation/independent_saved_output_review.json','results/class_patch_fresh_saved_review_v2_root_review.json','results/research_class_patch_dispersion/verification.json','results/research_class_patch_dispersion/comparison.json','results/research_class_patch_dispersion/independent_saved_output_review.json']
    for name,expected in artifacts.items():assert sha(name)==expected,name
    for name in names:artifacts[str(ROOT/name)]=sha(ROOT/name)
    assert sha(ROOT/names[2])=='bfbdcaa415abfa0eec440a4e2e3fb6ff0c8618084f9d8052e09028e1ed79603d'
    audit=json.loads((ROOT/'results/research_class_patch_dispersion/verification.json').read_text())
    result=json.loads((ROOT/'results/research_class_patch_dispersion/comparison.json').read_text())
    saved=json.loads((ROOT/'results/research_class_patch_dispersion/independent_saved_output_review.json').read_text())
    assert audit['passed'] and audit['actual_seed_heads_checked']==audit['actual_training_witnesses_checked']==24
    assert saved['review_passed'] and saved['gate_passers']==0 and not any(r['development_gate_passed'] for r in result['candidates'])
    protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'execution_entrypoint':'verify_class_patch_spatial_semivariance_model.py','code_sha256':{n:sha(ROOT/n) for n in code},'input_artifact_sha256':artifacts,'excluded_mutable_reporting_modules_before_closure':sorted(EXCLUDED),'classification_fits':0,'classification_scores':0,'actual_model_cases_planned':6,'engineering_optimizer_steps_planned':2,'external_test_evaluated':False,'purpose':'Actual256trainingimages per uniform/source x mean/globalvariance/spatial model/task/fullgradients/witness/restore cases, exact parent reductions and fixed sparse graph invariants; no classifier fit, score or backbone extraction.'}
    dump(out/'protocol.json',protocol)
    dump(out/'state.json',{'status':'frozen_before_engineering_proof','classification_fits':0,'classification_scores':0})
    env=dict(os.environ);env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
    with (out/'proof.log').open('x') as log:
        process=subprocess.Popen([sys.executable,'-u',str(ROOT/protocol['execution_entrypoint'])],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    dump(out/'launch_record.json',{'pid':process.pid,'protocol_sha256':sha(out/'protocol.json'),'classification_fits':0,'classification_scores':0})
    print(json.dumps({'pid':process.pid,'code_modules':len(code),'frozen_inputs':len(artifacts),'protocol_sha256':sha(out/'protocol.json')}),flush=True)
if __name__=='__main__':main()
