"""Verified development-only16position query/shared caches; no label-derived selection."""
import json
import numpy as np
from experiment import ROOT
from kernel_residual_experiments import sha
from register_patch_query_data import load_all as original_load

OUT=ROOT/'runs/register_query_selection_bank'
CACHE_MAP={mode+'16_'+suffix:mode+'_'+suffix for mode in ['query','shared'] for suffix in ['keys','values','indices','mass']}
INPUT_KEYS=['query_raw','postattention']+list(CACHE_MAP)


def load_all():
    assert json.loads((OUT/'state.json').read_text())['status']=='completed'
    proof=json.loads((OUT/'verification.json').read_text())
    review_path=ROOT/'results/register_query_selection_bank_root_review.json'
    review=json.loads(review_path.read_text());assert review['passed'] and len(review['saved_cpu_witnesses'])==6 and review['chunks_checked']==262
    assert review['input_artifact_sha256'][str(OUT/'verification.json')]==sha(OUT/'verification.json')
    assert proof['passed'] and proof['development_images']==2091 and proof['backbone_view_forwards']==530
    assert proof['classification_fits']==proof['classification_scores']==proof['classifier_probability_predictions']==proof['external_images_read']==0
    assert proof['all_six_original_arrays_reproduced_bitwise_for_every_image'] and proof['fresh_six_view_all_arrays_replay_bitwise']
    protocol=json.loads((OUT/'protocol.json').read_text());assert sha(OUT/'protocol.json')==proof['protocol_sha256']
    for name,expected in protocol['code_sha256'].items():assert sha(name)==expected,name
    for name,expected in proof['frozen_artifacts'].items():assert sha(name)==expected,name
    frame,features,original,tail,digest=original_load();assert digest==proof['manifest_sha256']==review['manifest_sha256']
    bank={key:original[key] for key in ['query_raw','postattention']}
    for key,name in CACHE_MAP.items():
        value=np.load(OUT/(name+'.npy'),mmap_mode='r')
        assert list(value.shape)==proof['shapes'][name] and value.dtype==np.dtype(proof['dtypes'][name])
        bank[key]=value
    return frame,features,bank,tail,digest
