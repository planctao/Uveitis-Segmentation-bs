"""Mode-scoped raw-image RAM activation loading; freeze complete cache contents before fitting."""
import json
import numpy as np
from experiment import ROOT,dump
from register_multiquery_data import load_all as load_source_features
from large_lastblock_delta_lora_cache_v2 import build_cache
from large_lastblock_delta_lora import make_templates

def load_for_run(mode,out):
    assert mode in ['original','fresh']
    frame,features,digest=load_source_features()
    ids=np.flatnonzero(frame.split.isin(['train','val']) if mode=='original' else frame.split.eq('train'))
    assert len(ids)==(2091 if mode=='original' else 1717)
    protocol=json.loads((ROOT/'runs/dinov3_large/protocol.json').read_text())
    cache,metadata=build_cache(frame,ids,protocol,independent_prefix_images=8)
    assert metadata['cache_storage_dtype']=='float32'
    freeze=out/'ram_activation_cache_before_any_fit.json';assert not freeze.exists()
    dump(freeze,{'mode':mode,'classification_fits_so_far':0,'manifest_sha256':digest,'metadata':metadata,
        'scope':'Actual read-only RAM token/rotary contents frozen before any fit; reconstruct from immutable producer/source image/backbone bytes, not a durable RAM artifact.'})
    if mode=='fresh':
        for values in features.values():values[~frame.split.eq('train').to_numpy()]=np.nan
    return frame,features,cache,make_templates(protocol),digest

def load_original_all():return load_for_run('original',ROOT/'runs/large_lastblock_delta_lora')
def load_fresh_all():return load_for_run('fresh',ROOT/'runs/large_lastblock_delta_lora_confirmation')
