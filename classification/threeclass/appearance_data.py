"""All development descriptors; every statistical fit remains training-patient only."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from experiment import ROOT,load_data,dump
from image_style_diagnostics import describe,NAMES
from kernel_residual_experiments import sha

OUT=ROOT/'runs/appearance_calibration'


def prepare():
    frame=pd.read_csv(ROOT/'metadata/manifest.csv');digest=sha(ROOT/'metadata/manifest.csv')
    prior=ROOT/'runs/image_style_diagnostics/training_descriptors.npz';train=np.load(prior)
    train_indices=np.flatnonzero(frame.split.eq('train'));valid=np.flatnonzero(frame.split.eq('val'))
    assert str(train['manifest_sha256'])==digest and np.array_equal(train['indices'],train_indices)
    assert np.array_equal(train['names'],NAMES)
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(describe,[(int(i),frame.iloc[i].cache_path) for i in valid],chunksize=16))
    assert [r[0] for r in rows]==valid.tolist()
    indices=np.r_[train_indices,valid];order=np.argsort(indices);indices=indices[order]
    assert np.array_equal(indices,np.arange(2091))
    values=np.r_[train['values'],np.array([r[1] for r in rows])][order]
    hashes=np.r_[train['cache_sha256'],np.array([r[2] for r in rows])][order]
    assert np.isfinite(values).all()
    np.savez_compressed(OUT/'development_descriptors.npz',indices=indices,values=values.astype('float32'),names=np.array(NAMES),
        manifest_sha256=np.array(digest),path=frame.iloc[indices].path.to_numpy(dtype=str),cache_sha256=hashes)
    for i in [0,108,517,1029,1717,2090]:
        _,expected,cache_hash=describe((i,frame.iloc[i].cache_path))
        assert np.allclose(values[i],expected,atol=1e-12,rtol=1e-12) and cache_hash==hashes[i]
    assert np.array_equal(values[train_indices],train['values'])
    dump(OUT/'feature_provenance.json',{'passed':True,'manifest_sha256':digest,'development_images':2091,'descriptor_names':NAMES,
        'source_training_descriptors_sha256':sha(prior),'feature_sha256':sha(OUT/'development_descriptors.npz'),
        'checks':['Alloriginal1717trainingdescriptorsreusedexactly;374validationdescriptorscomputedwithidenticalfixedimageformula',
            'Exactdevelopmentrow/pathalignmentandcachehashesretained;sixactualdescriptorsrecomputed',
            'No labels used for descriptor extraction; every subsequent mean/std fitted within currentfitpatients only',
            'Externalrawimagesexcluded; no external descriptors or predictions accessed'],
        'external_test_evaluated':False})
    print('Appearance descriptors prepared and verified for all2091developmentimages',flush=True)


def load_appearance():
    frame,features,digest=load_data('multiscale_vit')
    saved=np.load(OUT/'development_descriptors.npz');proof=json.loads((OUT/'feature_provenance.json').read_text())
    assert proof['passed'] and proof['feature_sha256']==sha(OUT/'development_descriptors.npz')
    indices=np.flatnonzero(frame.split.isin(['train','val']));assert np.array_equal(indices,saved['indices'])
    assert str(saved['manifest_sha256'])==digest and np.array_equal(frame.iloc[indices].path,saved['path'])
    style=np.full((len(frame),12),np.nan,dtype='float32');style[indices]=saved['values'];features['style']=style
    return frame,features,digest


if __name__=='__main__':prepare()
