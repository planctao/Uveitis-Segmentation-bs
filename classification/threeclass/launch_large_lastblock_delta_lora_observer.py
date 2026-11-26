"""Bind a scientific observer to exactly the actual worker from its launch record."""
import json,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
from experiment import ROOT,dump
from start_large_lastblock_delta_lora import sha

def main():
 out=ROOT/'runs/large_lastblock_delta_lora_observer';assert not (out/'protocol.json').exists()
 run=ROOT/'runs/large_lastblock_delta_lora';launch=json.loads((run/'launch_record.json').read_text())
 pid=launch['pid'];entrypoint='large_lastblock_delta_lora_experiments.py'
 command=Path('/proc')/str(pid)/'cmdline';assert str(ROOT/entrypoint).encode() in command.read_bytes().split(b'\0')
 protocol=json.loads((run/'protocol.json').read_text());assert sha(run/'protocol.json')==launch['protocol_sha256']
 for section in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
  for name,expected in protocol[section].items():assert sha(name)==expected,name
 old=(ROOT/'observe_class_patch_log_moment.py').read_text()
 expected=old.replace('class_patch_log_moment','large_lastblock_delta_lora').replace("fresh['new_seed_fits']==126","fresh['new_seed_fits']==108").replace("checked['actual_training_witnesses_checked']==90","checked['actual_training_witnesses_checked']==72").replace("['matched_mean','matched_variance']","['matched_token']")
 assert expected==(ROOT/'observe_large_lastblock_delta_lora.py').read_text()
 frozen=dict(protocol['frozen_artifacts']);frozen[str(run/'protocol.json')]=sha(run/'protocol.json')
 for name in ['verification_large_lastblock_delta_lora_startup.json','runs/large_lastblock_delta_lora_startup/protocol_preview.json','runs/large_lastblock_delta_lora_saved_review/protocol.json','runs/large_lastblock_delta_lora_fresh_saved_review/rule_before_outcome.json']:
  frozen[str(ROOT/name)]=sha(ROOT/name)
 science=dict(protocol['code_sha256']);science['launch_large_lastblock_delta_lora_observer.py']=sha(__file__)
 # Observer verifier accepts absolute code identities, unlike the original model runner.
 science={str(ROOT/name):digest for name,digest in science.items()}
 frozen_protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'worker_pid':pid,'execution_entrypoint':'observe_large_lastblock_delta_lora.py',
  'worker_entrypoint':entrypoint,'worker_actual_cmdline_verified':True,'code_sha256':science,'frozen_artifacts':frozen,
  'observer_structure':'Exact literal family/108/72/token-control transformation of completed log-moment observer. Prior terminal/binding logic coverage inherited, not new tests.',
  'reference_promotion':False,'external_test_evaluation':False}
 dump(out/'protocol.json',frozen_protocol)
 env=dict(os.environ);env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
 with (ROOT/'large_lastblock_delta_lora_observer.log').open('x') as log:
  process=subprocess.Popen([sys.executable,'-u',str(ROOT/frozen_protocol['execution_entrypoint'])],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 dump(out/'launch_record.json',{'pid':process.pid,'bound_worker_pid':pid,'protocol_sha256':sha(out/'protocol.json')})
 print(json.dumps({'pid':process.pid,'bound_worker_pid':pid,'protocol_sha256':sha(out/'protocol.json')}),flush=True)
if __name__=='__main__':main()
