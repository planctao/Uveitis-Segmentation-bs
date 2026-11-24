"""Persist the next development hypothesis and execute its complete experiment."""
import json
import fcntl
import os
import subprocess
import sys
import traceback
from experiment import ROOT,dump
from capacity_experiments import plan


def run():
    destination=ROOT/'runs/backbone_capacity'
    dump(destination/'protocol.json',{'hypothesis':'Twenty-one head-only structural ablations did not beat the frozen ViT-B development CV; test representational capacity and complementarity with public self-supervised DINOv2-L features.','configuration_plan':[{'feature':feature,'config':cfg} for feature,cfg in plan()],'selection':'All six configurations receive three seeds and three outer patient folds; inner patient holdout only for stopping. No external evaluation or thresholds chosen from external data.','preprocessing':'Same peripheral crop, aspect-preserving black padding and fixed manifest; 322 input is divisible by the 14-pixel patch size.','gpu_policy':'One frozen large backbone; small heads continue in the existing search process.'})
    try:
        for phase,script in [('download','download_dinov2_weights.py'),('extract','extract_dinov2_large.py'),('train_and_cv','capacity_experiments.py')]:
            dump(destination/'state.json',{'status':'running','phase':phase,'pid':os.getpid(),'external_test_evaluated':False})
            subprocess.run([sys.executable,'-u',str(ROOT/script)],cwd=ROOT,check=True)
        dump(destination/'state.json',{'status':'completed_predeclared_experiments','pid':os.getpid(),'external_test_evaluated':False})
    except Exception:
        dump(destination/'state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc(),'external_test_evaluated':False});raise


def main():
    destination=ROOT/'runs/backbone_capacity';destination.mkdir(parents=True,exist_ok=True)
    with (destination/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (destination/'runner.pid').write_text(str(os.getpid()))
        run()


if __name__=='__main__':main()
