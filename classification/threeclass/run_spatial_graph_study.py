"""Start graph comparison after the matching foreground controls are audited."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from experiment import ROOT,dump

OUT=ROOT/'runs/spatial_graph'


def main():
    with (OUT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(OUT/'pipeline.pid').write_text(str(os.getpid()))
        pid=int((ROOT/'runs/foreground_multiscale/pipeline.pid').read_text())
        dump(OUT/'pipeline_state.json',{'status':'waiting_for_foreground_controls','pid':os.getpid(),'source_pipeline_pid':pid})
        while True:
            path=Path('/proc')/str(pid)/'cmdline'
            if not path.exists() or b'run_foreground_study.py' not in path.read_bytes():break
            time.sleep(5)
        state=json.loads((ROOT/'runs/foreground_multiscale/pipeline_state.json').read_text())
        if state['status']!='completed':
            dump(OUT/'pipeline_state.json',{'status':'source_not_successful','pid':os.getpid(),'source_terminal_state':state});return
        try:
            dump(OUT/'pipeline_state.json',{'status':'running','pid':os.getpid()})
            subprocess.run([sys.executable,'-u',str(ROOT/'spatial_graph_experiments.py')],cwd=ROOT,check=True)
            dump(OUT/'pipeline_state.json',{'status':'completed','pid':os.getpid(),'external_test_evaluated':False})
        except BaseException:
            dump(OUT/'pipeline_state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc()});raise


if __name__=='__main__':main()
