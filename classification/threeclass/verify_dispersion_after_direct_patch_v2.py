"""Check predecessor decisions without launching or fitting a child process."""
from unittest.mock import patch
from experiment import ROOT, dump
from kernel_residual_experiments import sha
import continue_dispersion_after_direct_patch_v2 as bridge


def main():
    destination=ROOT/'verification_dispersion_after_direct_patch_v2.json'
    assert not destination.exists()
    cases=[]
    for name,state,live,expected in [
        ('negative_terminal',{'status':'completed_class_patch_evidence_decision_requires_next_exploration','pid':123},False,'negative'),
        ('positive_terminal',{'status':'confirmation_passed_requires_reference_review','pid':123},False,'positive'),
        ('live_same_observer',{'status':'waiting_register_patch_attention_coupling','pid':123},True,'wait'),
        ('changed_pid',{'status':'waiting_register_patch_attention_coupling','pid':124},True,'reject'),
        ('exited_observer',{'status':'waiting_register_patch_attention_coupling','pid':123},False,'reject'),
        ('failed_observer',{'status':'failed','pid':123},True,'reject'),
    ]:
        with patch('subprocess.Popen',side_effect=RuntimeError('Forbidden child process in proof')):
            try:actual=bridge.predecessor_status(state,123,live)
            except AssertionError:assert expected=='reject'
            else:assert actual==expected
        cases.append({'case':name,'expected':expected,'passed':True})
    assert bridge.predecessor_live(12660),'Actual original direct-patch observer must match its actual cmdline'
    assert not bridge.predecessor_live(999999999)
    dump(destination,{'passed':True,'cases':cases,
        'code_sha256':{n:sha(ROOT/n) for n in ['continue_dispersion_after_direct_patch.py','continue_dispersion_after_direct_patch_v2.py','verify_dispersion_after_direct_patch_v2.py']},
        'actual_predecessor_observer_pid_checked':12660,'actual_predecessor_cmdline_checked':True,'missing_pid_rejected':True,'classification_fits':0,'classification_scores':0,'child_processes_started':0,'external_images_read':0,
        'scope':'Six pure predecessor-decision checks only. The full queue runtime awaits a terminal predecessor and the actual formal study.'})
    print('Six predecessor decision cases passed; no launch or classifier fit.',flush=True)


if __name__=='__main__':main()
