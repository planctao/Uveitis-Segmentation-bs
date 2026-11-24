"""Explore higher resolution and aligned coarse/fine DINO region fusion."""
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
from experiment import ROOT,dump

def main():
    out=ROOT/'features';a=np.load(out/'vitb_320.npz');b=np.load(out/'vitb_448.npz')
    assert str(a['manifest_sha256'])==str(b['manifest_sha256']) and np.array_equal(a['path'],b['path'])
    arrays={k:np.concatenate([a[k],b[k]],axis=-1) for k in ['global','regions','mid','dispersion']}
    arrays.update(path=a['path'],manifest_sha256=a['manifest_sha256'])
    np.savez_compressed(out/'multiscale_vit.npz',**arrays)
    dump(out/'multiscale_vit.json',{'sources':['vitb_320','vitb_448'],'operation':'Concatenate aligned global and 3x3 regional channels across resolutions; no labels used','manifest_sha256':str(a['manifest_sha256']),'shapes':{k:list(v.shape) for k,v in arrays.items()}})
    reference=json.loads((ROOT/'runs/vitb_320/explore/selection.json').read_text())['best']['val']['macro_f1']
    results={}
    for feature in ['vitb_448','multiscale_vit']:
        for phase in ['baseline','explore']:
            subprocess.run([sys.executable,'-u',str(ROOT/'experiment.py'),'--features',feature,'--phase',phase],cwd=ROOT,check=True)
        selection=json.loads((ROOT/'runs'/feature/'explore/selection.json').read_text());best=selection['best']
        if best['val']['macro_f1']>reference:
            subprocess.run([sys.executable,'-u',str(ROOT/'crossvalidate.py'),'--features',feature],cwd=ROOT,check=True)
        results[feature]={'best':best,'improved_validation_macro_f1':best['val']['macro_f1']>reference}
        dump(ROOT/'runs/resolution_round.json',{'reference_macro_f1':reference,'results':results,'external_tests_evaluated':False})

if __name__=='__main__':main()
