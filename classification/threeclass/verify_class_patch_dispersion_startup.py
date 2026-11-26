"""Exercise startup rejection paths without starting fitting, scoring, or child processes."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from experiment import ROOT, dump
from kernel_residual_experiments import sha
import start_class_patch_dispersion as original
import start_class_patch_dispersion_confirmation as fresh
import verify_class_patch_dispersion_confirmation as selected
import review_class_patch_dispersion_outputs as reviewer


def main():
    destination=ROOT/'verification_class_patch_dispersion_startup.json'
    assert not destination.exists()
    cases=[]
    for name, module, fixture, action, error in [
        ('reject_duplicate_original',original,{'runs/class_patch_dispersion/protocol.json':{}},original.main,AssertionError),
        ('reject_active_predecessor',original,{'runs/class_query_patch_evidence/state.json':{'status':'training'}},original.main,AssertionError),
        ('reject_duplicate_fresh',fresh,{'runs/class_patch_dispersion_confirmation/protocol.json':{}},fresh.main,AssertionError),
        ('reject_unselected_fresh',fresh,{},fresh.main,FileNotFoundError),
        ('reject_unselected_formula_review',selected,{},selected.main,FileNotFoundError),
        ('reject_unstarted_saved_review',reviewer,{},lambda:reviewer.review('class_patch_dispersion'),AssertionError),
        ('reject_active_saved_review',reviewer,{'runs/class_patch_dispersion/state.json':{'status':'training'}},lambda:reviewer.review('class_patch_dispersion'),AssertionError),
    ]:
        with tempfile.TemporaryDirectory(prefix='coupling-startup-guard-') as directory:
            root=Path(directory)
            for file,content in fixture.items():
                path=root/file;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(content))
            before={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
            with patch.object(module,'ROOT',root),patch('subprocess.Popen',side_effect=RuntimeError('Forbidden launch in guard proof')):
                try:action()
                except error as exc:message=str(exc)
                else:raise AssertionError(name+' unexpectedly proceeded')
            after={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
            assert before==after,name
            cases.append({'case':name,'expected_rejection':error.__name__,'message':message,'no_process_launch_or_file_mutation':True})
    names=['start_class_patch_dispersion.py','start_class_patch_dispersion_confirmation.py',
        'verify_class_patch_dispersion_confirmation.py','review_class_patch_dispersion_outputs.py',
        'verify_class_patch_dispersion_startup.py']
    dump(destination,{'passed':True,'guard_cases':cases,'code_sha256':{name:sha(ROOT/name) for name in names},
        'classification_fits':0,'classification_scores':0,'model_predictions':0,'child_processes_started':0,
        'external_images_read':0,'scope':'Only rejected startup paths tested. Does not claim the unstarted original/fresh/full saved-output workflow has run. Actual prediction/restore/full-task engine checks are recorded separately.'})
    print('Seven duplicate/active/unselected guards rejected before file changes or process launches.',flush=True)


if __name__=='__main__':main()
