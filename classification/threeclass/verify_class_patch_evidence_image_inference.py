"""Frozen engineering checks on predefined development images, without classification scoring."""
import fcntl,json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from experiment import ROOT,KEYS,load_data,dump
from kernel_residual_experiments import sha
from large_token_data import load_all
from register_patch_query_features import BANK_KEYS
from research_class_patch_evidence_image_inference import FrozenPatchImageExtractor,adapter_probability,restore_reviewed_class_patch
from class_query_patch_evidence_experiments import predict
OUT=ROOT/'runs/class_patch_evidence_image_inference_check'
DEST=ROOT/'results/class_patch_evidence_image_inference_check.json'

def frozen(protocol):
    for section in ['code_sha256','frozen_artifacts','image_sha256']:
        for name,expected in protocol[section].items():assert sha(name)==expected,name

def status(value,**details):dump(OUT/'state.json',{'status':value,'pid':os.getpid(),**details,'external_test_evaluated':False})

def main_run():
    started=time.time();assert not DEST.exists();protocol=json.loads((OUT/'protocol.json').read_text());frozen(protocol);status('loading')
    frame,large,tokens,digest=load_all();_,base,other=load_data('multiscale_vit');assert digest==other==protocol['manifest_sha256']
    bank={k:np.load(ROOT/'runs/register_patch_query_bank/arrays'/(k+'.npy'),mmap_mode='r') for k in BANK_KEYS}
    features={**large,'side_tokens':tokens['registers']};extractor=FrozenPatchImageExtractor();batch_cases=[];single_cases=[];head_batch_forwards=0;independent_formula_batches=0;head_image_evaluations=0
    for indices in protocol['development_batches']:
        ids=np.asarray(indices,dtype='int64');assert len(ids)==8 and ids[0]%8==0 and np.array_equal(ids,np.arange(ids[0],ids[0]+8))
        assert frame.iloc[ids].split.isin(['train','val']).all();status('checking_matched_eight_image_batch',manifest_indices=indices)
        raw=extractor.prepare_images(frame.iloc[ids].path.tolist());cache=extractor.prepare_images(frame.iloc[ids].cache_path.tolist(),True)
        assert all(np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(raw,cache))
        live_base,live_large,live_bank=extractor.extract_patch_preprocessed(raw)
        cache_large={k:features[k][ids] for k in features};cache_bank={k:np.asarray(v[ids]) for k,v in bank.items()};feature_errors={}
        for branch,actual,expected in [('B',live_base,{k:base[k][ids] for k in KEYS}),('L',live_large,cache_large)]:
            for k in actual:
                assert np.allclose(actual[k],expected[k],**protocol['feature_tolerance']),(branch,k)
                feature_errors[branch+'_'+k]=float(np.abs(actual[k]-expected[k]).max())
        for k in BANK_KEYS:assert np.array_equal(live_bank[k],cache_bank[k]) and live_bank[k].dtype==cache_bank[k].dtype,k
        checks=[]
        for selected in protocol['checkpoint_cases']:
            model,source=restore_reviewed_class_patch(**selected)
            original=predict(model,source,features,bank,ids);manual=predict(model,source,features,bank,ids,independent=True)
            cached=adapter_probability(model,source,cache_large,cache_bank);live=adapter_probability(model,source,live_large,live_bank)
            head_batch_forwards+=3;independent_formula_batches+=1;head_image_evaluations+=4*len(ids)
            assert np.array_equal(cached,original),'API cached inputs must match original engine bitwise.'
            assert np.allclose(cached,manual,atol=2e-6,rtol=2e-6) and np.array_equal(cached.argmax(1),manual.argmax(1))
            assert np.allclose(live,cached,**protocol['matched_batch_probability_tolerance'])
            assert np.array_equal(live.argmax(1),cached.argmax(1))
            checks.append({**selected,'new_api_cached_probability_bitwise_matches_original_engine':True,
                'independent_probability_maximum_error':float(np.abs(cached-manual).max()),
                'raw_image_vs_cached_probability_maximum_error':float(np.abs(live-cached).max()),'raw_image_vs_cached_decisions_identical':True})
            del model;torch.cuda.empty_cache()
        batch_cases.append({'manifest_indices':indices,'raw_preprocessed_pixels_equal_stored_cache':True,
            'all_six_patch_arrays_bitwise_match_original_batch':True,'feature_maximum_errors':feature_errors,'checkpoint_checks':checks})
        print(json.dumps({'event':'direct_patch_matched_batch_checked','indices':indices,'checkpoints':len(checks)}),flush=True)
    for index in protocol['validation_indices']:
        assert frame.iloc[index].split=='val';status('checking_single_image_request',manifest_index=index)
        _,live_large,live_bank=extractor.extract_patch_preprocessed(extractor.prepare_images([frame.iloc[index].path]))
        cache_large={k:v[index:index+1] for k,v in features.items()};cache_bank={k:np.asarray(v[index:index+1]) for k,v in bank.items()}
        errors={}
        for k in live_large:
            assert np.allclose(live_large[k],cache_large[k],**protocol['feature_tolerance']),k
            errors[k]=float(np.abs(live_large[k]-cache_large[k]).max())
        overlaps=[len(set(live_bank['patch_indices'][0,v,h].tolist()) & set(cache_bank['patch_indices'][0,v,h].tolist()))/64 for v in range(2) for h in range(16)]
        checks=[]
        for selected in protocol['checkpoint_cases']:
            model,source=restore_reviewed_class_patch(**selected)
            live=adapter_probability(model,source,live_large,live_bank);cached=adapter_probability(model,source,cache_large,cache_bank)
            head_batch_forwards+=2;head_image_evaluations+=2
            assert np.allclose(live,cached,**protocol['single_image_probability_tolerance']),(selected,index,float(np.abs(live-cached).max()))
            assert np.array_equal(live.argmax(1),cached.argmax(1)),(selected,index)
            checks.append({**selected,'maximum_single_vs_cached_probability_error':float(np.abs(live-cached).max()),'decisions_identical':True})
            del model;torch.cuda.empty_cache()
        single_cases.append({'validation_index':index,'all32head_patch_overlap_fractions':overlaps,
            'minimum_head_patch_overlap_fraction':min(overlaps),'large_feature_maximum_errors':errors,'checkpoint_checks':checks})
        print(json.dumps({'event':'direct_patch_single_image_checked','index':index,'checkpoints':len(checks)}),flush=True)
    frozen(protocol)
    dump(DEST,{'passed':True,'protocol_sha256':sha(OUT/'protocol.json'),'manifest_sha256':digest,
        'matched_batch_cases':batch_cases,'single_image_cases':single_cases,'matched_batch_images':24,'single_image_requests':3,
        'reviewed_checkpoints':len(protocol['checkpoint_cases']),'includes_all_three_selected_validation_seed_heads':True,
        'production_head_batch_forwards':head_batch_forwards,'independent_formula_batch_forwards':independent_formula_batches,'head_image_evaluations':head_image_evaluations,
        'base_backbone_view_forwards':12,'large_backbone_view_forwards':12,'total_backbone_view_forwards':24,
        'new_model_fits':0,'new_candidate_scores':0,'external_images_read':0,'external_test_evaluated':False,
        'feature_tolerance':protocol['feature_tolerance'],'matched_batch_probability_tolerance':protocol['matched_batch_probability_tolerance'],'single_image_probability_tolerance':protocol['single_image_probability_tolerance'],
        'scope':'Actual raw-image numerical engineering checks on the same predefined24developmentimages and3singlevalidationrequests as earlier inference proofs. Includes all3selectedclassqueryvalidationheads plus oneuniform/sourcequery/contrast checkpoint. No accuracy, ranking, gain, full validation or external inference claim; no package/reference promotion. Exact top64 equality only at original8imagebatchshape.',
        'seconds':time.time()-started,'code_sha256':protocol['code_sha256']})
    status('completed',passed=True);print('Direct patch original-image engineering checks completed.',flush=True)

def main():
    torch.set_num_threads(4)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main_run()
        except BaseException:status('failed',error=traceback.format_exc());raise
if __name__=='__main__':main()
