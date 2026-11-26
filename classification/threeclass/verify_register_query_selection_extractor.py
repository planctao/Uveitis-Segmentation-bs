"""Two original-batch sourceview probes before freezing new full cache extraction."""
import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from experiment import ROOT
from extract_dinov3_large import Images,make_model
from extract_register_query_selection_bank import extract,independent
from register_patch_query_features import BANK_KEYS
from kernel_residual_experiments import sha


def main():
    torch.set_num_threads(4)
    destination=ROOT/'verification_register_query_selection_extractor.json';assert not destination.exists()
    frame=pd.read_csv(ROOT/'metadata/manifest.csv');ids=np.arange(8);assert frame.iloc[ids].split.isin(['train','val']).all()
    protocol=ROOT/'runs/dinov3_large/protocol.json'
    model=make_model(json.loads(protocol.read_text()))
    bank={name:np.load(ROOT/'runs/register_patch_query_bank/arrays'/(name+'.npy'),mmap_mode='r') for name in BANK_KEYS}
    _,a,b=next(iter(DataLoader(Images(frame,ids),batch_size=8)))
    cases=[]
    for view,tensor in enumerate([a,b]):
        arrays,witness,error=extract(model,tensor.cuda(),ids,view,bank)
        cpu_error=independent(witness)
        assert arrays['query_keys'].shape==(8,16,4,16,64) and arrays['shared_keys'].shape==(8,16,16,64)
        cases.append({'view':view,'images':8,'all_six_original_arrays_bitwise_equal':True,'independent_cpu_maximum_error':cpu_error,'conditional_times_mass_maximum_error':error})
    result={'passed':True,'created_at_utc':datetime.now(timezone.utc).isoformat(),'cases':cases,
        'manifest_sha256':sha(ROOT/'metadata/manifest.csv'),'source_protocol_sha256':sha(protocol),
        'code_sha256':{name:sha(ROOT/name) for name in ['extract_register_query_selection_bank.py','verify_register_query_selection_extractor.py']},
        'backbone_view_forwards':2,'classification_fits':0,'classification_scores':0,'classifier_probability_predictions':0,'external_images_read':0,
        'scope':'Two fixed8imageoriginalsourcebatchview preflight; extra2backboneforwards. Full530forward/all2091image extraction and rootreview stillrequired.'}
    with destination.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({'passed':True,'sha256':sha(destination),'cases':cases}))


if __name__=='__main__':main()
