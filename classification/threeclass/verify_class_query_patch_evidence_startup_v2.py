"""Versioned startup recovery: check rejection guards and construct the actual success protocol without launching."""
import io,json,os,tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
from experiment import ROOT,dump
from kernel_residual_experiments import sha
import start_class_query_patch_evidence_v2 as starter


def main():
    destination=ROOT/'verification_class_query_patch_evidence_startup_v2.json'
    assert not destination.exists()
    cases=[]
    for name,fixture in [
        ('reject_duplicate_original',{'runs/class_query_patch_evidence/protocol.json':{}}),
        ('reject_active_predecessor',{'runs/register_patch_attention_coupling/state.json':{'status':'training'}}),
    ]:
        with tempfile.TemporaryDirectory(prefix='class-patch-starter-v2-') as directory:
            root=Path(directory)
            for file,content in fixture.items():
                p=root/file;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(content))
            before={str(p):sha(p) for p in root.rglob('*') if p.is_file()}
            with patch.object(starter,'ROOT',root),patch('subprocess.Popen',side_effect=RuntimeError('No launch allowed')):
                try:starter.main()
                except AssertionError:pass
                else:raise AssertionError(name)
            assert before=={str(p):sha(p) for p in root.rglob('*') if p.is_file()}
        cases.append({'case':name,'passed':True})
    hypothesis=ROOT/'results/next_class_query_patch_evidence_hypothesis.json';before=sha(hypothesis)
    assert not (ROOT/'runs/class_query_patch_evidence/protocol.json').exists()
    buffer=io.StringIO()
    with patch.dict(os.environ,{'CLASS_PATCH_PREFLIGHT_ONLY':'1'}),patch('subprocess.Popen',side_effect=RuntimeError('No launch allowed')),redirect_stdout(buffer):
        protocol=starter.main()
    prereg=json.loads((ROOT/'results/class_query_patch_evidence_preregistration_before_attention_coupling_outcome.json').read_text())
    for target,source in [('training','fixed_training'),('configuration_plan','configuration_plan'),('architecture','architecture_plan'),('selection','original_selection'),('confirmation_plan','confirmation_plan'),('fixed_fusions','fixed_fusions')]:assert protocol[target]==prereg[source]
    assert protocol['planned_new_seed_fits']==48 and protocol['predeclared_scores']==8
    assert 'no automatic external scoring or reference promotion' in protocol['external_test_policy']
    assert sha(hypothesis)==before and not (ROOT/'runs/class_query_patch_evidence/protocol.json').exists()
    assert not (ROOT/'runs/class_query_patch_evidence/state.json').exists()
    preview=ROOT/'runs/class_query_patch_evidence_startup_v2_preflight/protocol_preview.json'
    dump(preview,protocol)
    names=['start_class_query_patch_evidence.py','start_class_query_patch_evidence_v2.py','verify_class_query_patch_evidence_startup_v2.py']
    dump(destination,{'passed':True,'guard_cases':cases,'success_path_protocol_constructed_without_launch':True,
        'success_path_output':json.loads(buffer.getvalue()),'protocol_preview':str(preview.relative_to(ROOT)),'protocol_preview_sha256':sha(preview),
        'original_preregistration_fields_unchanged':True,'hypothesis_bytes_unchanged':True,
        'code_sha256':{n:sha(ROOT/n) for n in names},'classification_fits':0,'classification_scores':0,
        'model_predictions':0,'child_processes_started':0,'external_images_read':0,
        'scope':'Actual protocol-construction success path with all real completed predecessor/model/engine checks; Popen forbidden and no formal protocol/state written. Full classifier and observer workflows await their actual run.'})
    print('Version2 actual success-path protocol and two rejection guards passed; no classifier fit or process launch.',flush=True)


if __name__=='__main__':main()
