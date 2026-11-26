"""Freeze and launch label-free patch KV extraction after feasibility passes."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime,timezone
import pandas as pd
from experiment import ROOT,dump
from kernel_residual_experiments import sha
from extract_register_patch_query_bank import SHAPES,DTYPES
import numpy as np


def main():
    out=ROOT/'runs/register_patch_query_bank';assert not (out/'protocol.json').exists()
    proofs=[ROOT/'verification_register_patch_query_feasibility.json',ROOT/'verification_register_patch_cache_oracle.json']
    for proof_path in proofs:
        proof=json.loads(proof_path.read_text());assert proof['passed'] and proof['classification_fits']==proof['classification_scores']==proof['external_images_read']==0
        for name,expected in proof['code_sha256'].items():assert sha(ROOT/name)==expected
    state=json.loads((ROOT/'runs/register_conditional_readout/state.json').read_text());assert state['status']=='completed' and state['passing_development_gates']==0
    verification=json.loads((ROOT/'results/research_register_conditional_readout/verification.json').read_text());assert verification['passed'] and verification['actual_seed_heads_checked']==verification['actual_training_witnesses_checked']==48
    from register_conditional_readout_experiments import verify_frozen
    prior=json.loads((ROOT/'runs/register_conditional_readout/protocol.json').read_text());verify_frozen(prior)
    source_path=ROOT/'runs/dinov3_large/protocol.json';source=json.loads(source_path.read_text());weight=Path(source['pretrained_weights']['storage'])
    assert sha(weight)==source['pretrained_weights']['sha256']
    estimated=sum(int(np.prod(SHAPES[key]))*np.dtype(DTYPES[key]).itemsize for key in SHAPES)
    assert shutil.disk_usage('/autodl-fs/data').free>estimated+1536*1024**2
    frame=pd.read_csv(ROOT/'metadata/manifest.csv');development=frame[frame.split.isin(['train','val'])];assert len(development)==2091
    artifacts=set(proofs+[source_path,weight,ROOT/'metadata/manifest.csv',ROOT/'verification_large_token_feature_bank.json'])
    code=['register_patch_query_features.py','register_patch_query.py','audit_register_patch_query.py','verify_register_patch_query_feasibility.py','verify_register_patch_cache_oracle.py','extract_register_patch_query_bank.py','start_register_patch_query_bank.py','extract_dinov3_large.py']
    storage=Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124/register_patch_query_bank');storage.mkdir(parents=True,exist_ok=True)
    if not out.exists():out.symlink_to(storage,target_is_directory=True)
    assert out.resolve()==storage.resolve()
    protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':sha(ROOT/'metadata/manifest.csv'),'purpose':'Label-free DINO-L finalregister top64 spatialKV cache for anchoredqueryadaptation; no classification fit/score.',
        'development_rows':2091,'scales':[320,448],'heads':16,'head_width':64,'patches_per_head':64,'batch_size':8,
        'shapes':{key:list(value) for key,value in SHAPES.items()},'dtypes':DTYPES,'estimated_array_bytes':estimated,
        'selection':'Stable descending originalregistermean attention across4registerqueries, fullprefix+patch softmax, then onlypatchpositions selectable. Deterministic tie index order. Postrotarykeys/rawvalues retained; rawq beforeqnorm (source qnorm/knorm Identity).',
        'numerics':'queryinput/preMLPstate float32 to avoid observed source residualoverflow; K/V/rawquery originalfloat16; downstream queryadapter selectedattention/pooldifference float64 then projection/tail/head float32. Counterfactualtailbaseline subtraction makes correctionzero atzeroB. No fullpatchLoRAequivalenceclaim.',
        'cache_image_sha256':{path:sha(path) for path in development.cache_path},'frozen_artifacts':{str(path):sha(path) for path in sorted(artifacts)},'code_sha256':{name:sha(ROOT/name) for name in code},'classification_fits':0,'classification_scores':0,'external_images_read':0}
    dump(out/'protocol.json',protocol);dump(out/'state.json',{'status':'frozen_before_extraction','classification_fits':0,'classification_scores':0,'external_images_read':0})
    env=dict(os.environ);env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
    with (ROOT/'register_patch_query_bank.log').open('a') as log:process=subprocess.Popen([sys.executable,'-u',str(ROOT/'extract_register_patch_query_bank.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    hypothesis=ROOT/'results/next_register_patch_query_hypothesis.json';a=json.loads(hypothesis.read_text());dump(hypothesis,{**a,'status':'extracting_label_free_bank_before_model_preflight','feature_protocol':'runs/register_patch_query_bank/protocol.json','feature_protocol_sha256':sha(out/'protocol.json'),'feature_pid':process.pid,'feasibility_preflight':'verification_register_patch_query_feasibility.json','cache_oracle':'verification_register_patch_cache_oracle.json'})
    print(json.dumps({'pid':process.pid,'estimated_array_bytes':estimated,'protocol_sha256':sha(out/'protocol.json'),'classification_fits':0,'classification_scores':0}),flush=True)


if __name__=='__main__':main()
