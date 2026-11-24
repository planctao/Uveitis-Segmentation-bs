"""Development-only large backbone features and predeclared base/large fusions."""
import json
from pathlib import Path
import numpy as np
from experiment import KEYS


def load_derived(root,frame,digest,feature):
    if feature not in ['large320','large_dual','base_dual_large_dual','base_dual_large448']:
        raise ValueError('Unknown fixed DINOv3-L feature arm')
    protocol=json.loads((root/'runs/dinov3_large/protocol.json').read_text())
    path=Path(protocol['feature_storage']);large=np.load(path)
    indices=np.flatnonzero(frame.split.isin(['train','val']))
    assert np.array_equal(indices,np.arange(2091)) and np.array_equal(large['indices'],indices)
    assert str(large['manifest_sha256'])==digest and np.array_equal(large['path'],frame.iloc[indices].path)
    base=None
    if feature.startswith('base_dual'):
        base=np.load(root/'features/multiscale_vit.npz')
        assert str(base['manifest_sha256'])==digest and np.array_equal(base['path'],frame.path)
    result={}
    for key in KEYS:
        values=large[key].astype('float32')
        if feature=='large320':values=values[...,:1024]
        elif feature=='base_dual_large448':values=values[...,1024:]
        if base is not None:values=np.concatenate([base[key][indices].astype('float32'),values],axis=-1)
        complete=np.full((len(frame),*values.shape[1:]),np.nan,dtype='float32');complete[indices]=values
        assert np.isfinite(complete[indices]).all() and np.isnan(complete[2091:]).all()
        result[key]=complete
    return result
