"""Test the observer's actual wait loop, including the known initial-state PID race."""
import ast
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from experiment import ROOT,dump
from kernel_residual_experiments import sha
import continue_class_patch_dispersion as observer


class ReachedWait(Exception):pass


def main():
    destination=ROOT/'verification_class_patch_dispersion_continuation.json'
    assert not destination.exists()
    code=ROOT/'continue_class_patch_dispersion.py'
    tree=ast.parse(code.read_text())
    assert not any(isinstance(n,ast.Attribute) and n.attr=='Popen' for n in ast.walk(tree))
    assert 'start_class_patch_dispersion.py' not in code.read_text()
    cases=[]
    for name,state,live,expected in [
        ('completed',{'status':'completed'},False,'completed'),
        ('failed',{'status':'failed'},True,'reject'),
        ('startup_missing_pid',{'status':'frozen_before_training'},True,'wait'),
        ('same_live_pid',{'status':'training','pid':123},True,'wait'),
        ('changed_pid',{'status':'training','pid':124},True,'reject'),
        ('missing_pid_after_startup',{'status':'training'},True,'reject'),
        ('exited_worker',{'status':'training','pid':123},False,'reject'),
    ]:
        with tempfile.TemporaryDirectory(prefix='class-patch-observer-proof-') as directory:
            root=Path(directory);folder=root/'runs/example';folder.mkdir(parents=True)
            (folder/'state.json').write_text(json.dumps(state));events=[]
            with patch.object(observer,'ROOT',root),patch.object(observer,'live',return_value=live),patch.object(observer,'status',side_effect=lambda *args,**kwargs:events.append((args,kwargs))),patch.object(observer.time,'sleep',side_effect=ReachedWait):
                try:
                    result=observer.wait('example','example.py',123)
                except ReachedWait:assert expected=='wait' and events[-1][1]['worker_pid']==123
                except AssertionError:assert expected=='reject'
                else:assert expected=='completed' and result==state
            cases.append({'case':name,'expected':expected,'passed':True})
    prior=ROOT/'verification_query_selection_continuation_v2.json'
    assert json.loads(prior.read_text())['passed']
    dump(destination,{'passed':True,'cases':cases,'observer_has_no_original_worker_launch_path':True,
        'code_sha256':{name:sha(ROOT/name) for name in ['continue_class_patch_dispersion.py','verify_class_patch_dispersion_continuation.py','continue_query_selection_after_mass_v2.py']},
        'reused_identity_proof':str(prior.relative_to(ROOT)),'reused_identity_proof_sha256':sha(prior),
        'classification_fits':0,'classification_scores':0,'external_images_read':0,'child_processes_started':0,
        'reused_wait_identity_rule':'Use the already-proven missing-startup-PID rule; no worker restart path.',
        'scope':'Seven actual wait-loop cases only; the full original/completion/confirmation queue still requires its actual formal run.'})
    print('Seven actual observer wait-loop cases passed; no original-worker relaunch path.',flush=True)


if __name__=='__main__':main()
