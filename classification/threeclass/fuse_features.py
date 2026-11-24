"""Concatenate aligned DINO features without accessing labels."""
import json
import numpy as np
from experiment import ROOT

def main():
    root=ROOT/'features';a=np.load(root/'convnext_320.npz');b=np.load(root/'vitb_320.npz')
    assert str(a['manifest_sha256'])==str(b['manifest_sha256'])
    assert np.array_equal(a['path'],b['path'])
    arrays={k:np.concatenate([a[k],b[k]],axis=-1) for k in ['global','regions','mid','dispersion']}
    arrays.update(path=a['path'],manifest_sha256=a['manifest_sha256'])
    np.savez_compressed(root/'fusion_320.npz',**arrays)
    (root/'fusion_320.json').write_text(json.dumps({'sources':['convnext_320','vitb_320'],'operation':'Concatenate corresponding feature channels; all rows checked for identity.','manifest_sha256':str(a['manifest_sha256']),'shapes':{k:list(v.shape) for k,v in arrays.items()}},indent=2))

if __name__=='__main__':main()
