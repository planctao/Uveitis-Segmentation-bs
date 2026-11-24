"""Ordered download, extraction, independent feature verification and fixed fits."""
import fcntl
import os
import subprocess
import sys
import traceback
from experiment import ROOT,dump

OUT=ROOT/'runs/dinov3_large'


def main():
    with (OUT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(OUT/'pipeline.pid').write_text(str(os.getpid()))
        for script in ['download_dinov3_large.py','extract_dinov3_large.py','verify_dinov3_large.py','dinov3_large_experiments.py']:
            dump(OUT/'pipeline_state.json',{'status':'running','pid':os.getpid(),'step':script,'external_test_evaluated':False})
            try:subprocess.run([sys.executable,'-u',str(ROOT/script)],cwd=ROOT,check=True)
            except BaseException:
                dump(OUT/'pipeline_state.json',{'status':'failed','pid':os.getpid(),'step':script,'error':traceback.format_exc()});raise
        dump(OUT/'pipeline_state.json',{'status':'completed','pid':os.getpid(),'external_test_evaluated':False})


if __name__=='__main__':main()
