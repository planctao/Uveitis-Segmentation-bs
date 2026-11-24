"""Check every extracted shard, data fusion, and actual image feature recomputation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment import ROOT,dump
from extract_dinov3_large import OUT,KEYS,Images,make_model,forward_features,fingerprint
from dinov3_large_data import load_derived


@torch.inference_mode()
def main():
    torch.set_num_threads(4);torch.manual_seed(20261124)
    protocol=json.loads((OUT/'protocol.json').read_text());digest=protocol['manifest_sha256']
    path=Path(protocol['feature_storage']);meta=json.loads(path.with_suffix('.json').read_text());saved=dict(np.load(path))
    frame=pd.read_csv(ROOT/'metadata/manifest.csv');indices=np.flatnonzero(frame.split.isin(['train','val']))
    assert meta['strict_loading'] and meta['external_raw_images_read']==0 and meta['development_images']==2091
    assert meta['pretrained_weights']==protocol['pretrained_weights']
    assert str(saved['manifest_sha256'])==digest and np.array_equal(saved['indices'],indices)
    for key in saved:
        value=saved[key];m=meta['arrays'][key]
        assert value.shape==tuple(m['shape']) and str(value.dtype)==m['dtype'] and fingerprint(value)==m['value_sha256']
    rows=[]
    for shard in meta['shards']:
        z=np.load(shard['path']);ids=z['indices'];assert ids.tolist()==shard['indices']
        assert str(z['manifest_sha256'])==digest
        for key in KEYS:
            assert fingerprint(z[key])==shard['arrays'][key] and np.array_equal(z[key],saved[key][ids])
        rows.extend(ids.tolist())
    assert rows==indices.tolist() and len(meta['shards'])==262
    base=np.load(ROOT/'features/multiscale_vit.npz')
    for arm in protocol['configuration_plan']:
        feature=arm['feature'];derived=load_derived(ROOT,frame,digest,feature)
        for key in KEYS:
            expected=saved[key]
            if feature=='large320':expected=expected[...,:1024]
            if feature=='base_dual_large448':expected=expected[...,1024:]
            if feature.startswith('base_dual'):expected=np.concatenate([base[key][indices],expected],axis=-1)
            assert np.array_equal(derived[key][indices],expected.astype('float32')) and np.isnan(derived[key][2091:]).all()
    model=make_model(protocol);mid={};hook=model.blocks[15].register_forward_hook(lambda module,inputs,output:mid.update(tokens=output))
    chosen=[0,352,887,1501,1800,2090];dataset=Images(frame,chosen);errors=[]
    for j,index in enumerate(chosen):
        actual_index,a,b=dataset[j];assert actual_index==index
        views=[forward_features(model,image[None].cuda(),mid) for image in [a,b]]
        error={}
        for key in KEYS:
            value=np.concatenate([view[key] for view in views],axis=-1)[0].astype('float32');expected=saved[key][index].astype('float32')
            difference=float(np.abs(value-expected).max());error[key]=difference
            assert np.allclose(value,expected,atol=.05,rtol=.005),(index,key,difference)
        errors.append({'development_index':index,'max_abs_feature_difference':error})
    hook.remove()
    dump(ROOT/'verification_dinov3_large.json',{'passed':True,'manifest_sha256':digest,'weights_sha256':protocol['pretrained_weights']['sha256'],
        'development_images_checked':2091,'shards_checked':262,'actual_images_recomputed':errors,
        'checks':['PinnedpublicweightcontentSHA256 and strict1024D24blockstate verified',
            'Allbankarrayfingerprints and every262shard matched exactly to2091developmentrows',
            'All4predeclaredfeaturelayouts independently reconstructed; allexternalrowsNaN',
            'Sixactualdevelopmentimages atboth320/448 independently reloadedandforwarded through frozenbackbone',
            'CLS/3x3region/block15mid/patchstdfeatures match savedAMPfloat16within statednumericaltolerances']})
    print('DINOv3-L complete bank/shard/fusion and actual image recomputation checks passed',flush=True)


if __name__=='__main__':main()
