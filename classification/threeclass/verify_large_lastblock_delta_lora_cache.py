"""Actual training-image cache/prefix/resumed-block engineering, without classifier fitting."""
import hashlib,json,os,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment import ROOT,dump
from large_lastblock_delta_lora_cache import build_cache

OUT=ROOT/'runs/large_lastblock_delta_lora_cache_preflight'
DEST=ROOT/'verification_large_lastblock_delta_lora_cache.json'
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def verify(protocol):
    for section in ['code_sha256','input_artifact_sha256','package_implementation_sha256']:
        for name,expected in protocol[section].items():
            path=Path(name);assert sha(path if path.is_absolute() else ROOT/path)==expected,name
def main():
    assert not DEST.exists(),'Preserve observed proof outputs; version recovery.'
    started=time.time();torch.set_num_threads(4)
    protocol=json.loads((OUT/'protocol.json').read_text());verify(protocol)
    frame=pd.read_csv(ROOT/'metadata/manifest.csv')
    fit=np.flatnonzero(frame.split.eq('train'))
    ids=fit[np.random.default_rng(20294400).permutation(len(fit))[:256]]
    assert ids.tolist()==protocol['indices'] and frame.iloc[ids].split.eq('train').all()
    parent=json.loads((ROOT/'runs/dinov3_large/protocol.json').read_text())
    cache,metadata=build_cache(frame,ids,parent)
    assert len(cache['indices'])==256 and metadata['independent_prefix_view_checks']==16
    assert metadata['full_backbone_view_evaluations']==metadata['float32_lastblock_view_evaluations']==512
    assert metadata['all_numpy_cache_arrays_read_only']
    verify(protocol)
    dump(DEST,{'passed':True,'seconds':time.time()-started,'manifest_sha256':sha(ROOT/'metadata/manifest.csv'),
        'protocol_sha256':sha(OUT/'protocol.json'),'code_sha256':protocol['code_sha256'],
        'input_artifact_sha256':protocol['input_artifact_sha256'],'package_implementation_sha256':protocol['package_implementation_sha256'],
        'metadata':metadata,'classification_fits':0,'classification_scores':0,'external_raw_images_read':0,
        'scope':'256 actual training images,512 full-backbone views,16 independent prefix checks and512 float32 final-block resumptions. No classifier task/gradient/anchor/control/optimizer/restore or accuracy evidence yet. RAM arrays intentionally discarded at process exit; immutable producer/input/content hashes permit reconstruction.'})
    print(json.dumps({'passed':True,'actual_training_images':256,'full_backbone_views':512,
        'maximum_bridge_error':max(r['maximum_absolute_error'] for r in metadata['bridge']),
        'maximum_bridge_relative_l2_error':max(r['relative_l2_error'] for r in metadata['bridge']),
        'cache_bytes':sum(r['bytes'] for r in metadata['arrays'].values()),
        'classification_fits':0,'classification_scores':0}),flush=True)
if __name__=='__main__':
    dump(OUT/'state.json',{'status':'running_cache_bridge_engineering','pid':os.getpid(),'classification_fits':0,'classification_scores':0})
    try:
        main()
        dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'classification_fits':0,'classification_scores':0})
    except BaseException:
        dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'traceback':traceback.format_exc(),'classification_fits':0,'classification_scores':0})
        raise
