"""Run a fixed high-resolution image experiment with explicit matched controls."""
import fcntl
import json
import os
import subprocess
import sys
import traceback
from experiment import ROOT,dump


def main():
    out=ROOT/'runs/raw_highres';out.mkdir(parents=True,exist_ok=True)
    with (out/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(out/'runner.pid').write_text(str(os.getpid()))
        reference=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())
        dump(out/'protocol.json',{'hypothesis':'The fixed 448-pixel cache may discard useful fine image information; test genuine 768-pixel inputs from original images against upsampling the existing cache to the same size.','development_evidence':'48 of 48 sampled training originals exceed 768 pixels; typical original size 2600x2048, while all sampled caches are 448x448.','arms':['original-to-768 single view','cache448-upsample-to768 single view','existing320+448 plus original-to768','existing320+448 plus cache448-upsample-to768'],'matched_controls':'Identical fixed 5 percent crop, aspect padding, JPEG quality95/subsampling0, DINOv3-B weights, input size, class head configuration and seeds. The high-resolution source differs.','head_configuration':reference['selected']['config'],'manifest_sha256':reference['manifest_sha256'],'evaluation':'All four configurations receive three seeds and three outer patient folds, with separate inner early-stopping patients. Passing candidates need repeated-patient confirmation before any external evaluation.','external_test_policy':'Feature extraction is frozen and label-independent; no external outcomes are evaluated by this pipeline.'})
        try:
            for phase,script in [('verify_inputs','verify_raw_highres.py'),('extract_paired_features','extract_raw_highres.py'),('train_and_cv','highres_experiments.py')]:
                dump(out/'state.json',{'status':phase,'pid':os.getpid(),'external_test_evaluated':False})
                subprocess.run([sys.executable,'-u',str(ROOT/script)],cwd=ROOT,check=True)
            dump(out/'state.json',{'status':'completed_development_experiments','pid':os.getpid(),'configurations':4,'external_test_evaluated':False})
        except BaseException:
            dump(out/'state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc(),'external_test_evaluated':False});raise


if __name__=='__main__':main()
