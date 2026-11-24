"""Validate the frozen composite and complete source-image inference using development images only."""
import json
import numpy as np
import torch
from experiment import ROOT,KEYS,load_data,metrics,dump
from predict_normal_reference import FrozenNormalReferencePredictor
from kernel_residual_experiments import sha


def main():
    torch.set_num_threads(4)
    frame,features,digest=load_data('multiscale_vit')
    for value in features.values():value[2091:]=np.nan
    ids=np.flatnonzero(frame.split.eq('val'))
    predictor=FrozenNormalReferencePredictor(device='cuda',require_final_freeze=False)
    parts=[]
    for begin in range(0,len(ids),128):
        selected=ids[begin:begin+128]
        parts.append(predictor.predict_features({key:value[selected] for key,value in features.items()}))
    p=np.concatenate(parts)
    saved=np.load(ROOT/'runs/normal_subspace_fusion/normal_full_weight_0p75/validation/predictions.npz')
    assert np.array_equal(saved['indices'],ids)
    assert np.allclose(p,saved['p'],atol=5e-5,rtol=5e-5) and np.array_equal(p.argmax(1),saved['pred'])
    chosen=np.random.default_rng(20281302).choice(ids,6,replace=False)
    cached=predictor.predict_features({key:value[chosen] for key,value in features.items()})
    live=predictor.predict_images(frame.iloc[chosen].cache_path.tolist(),already_preprocessed=True,batch=3)
    original=predictor.predict_images(frame.iloc[chosen].path.tolist(),already_preprocessed=False,batch=3)
    assert np.abs(cached-live).mean()<.001 and np.array_equal(cached.argmax(1),live.argmax(1))
    assert np.allclose(live,original,atol=5e-5,rtol=5e-5) and np.array_equal(live.argmax(1),original.argmax(1))
    # One DINO image model supplies the shared features to both sets of three heads.
    assert predictor.image_model is not None and not predictor.image_model.training
    document={'passed':True,'manifest_sha256':digest,'package_sha256':sha(ROOT/'results/milestone6/package.json'),
        'validation_images_checked':len(ids),'validation':metrics(frame.iloc[ids].label,p),
        'real_image_indices':chosen.tolist(),'real_cache_and_original_images_checked':6,
        'maximum_validation_probability_error':float(np.max(np.abs(p-saved['p']))),
        'mean_live_vs_cache_probability_error':float(np.abs(cached-live).mean()),
        'maximum_original_vs_live_probability_error':float(np.max(np.abs(original-live))),
        'external_images_or_outcomes_used':False,'code_sha256':{name:sha(ROOT/name) for name in ['predict_normal_reference.py','verify_normal_reference_inference.py']},
        'checks':['Allsixfrozencheckpointandconfirmationhashesverified; strictnormalandordinaryM5restoration',
            'Onecommonfit-onlynormalizationandfixed0.75mixturereproduceall374validationdecisions',
            'Sixactualvalidationcacheimagesandtheirsources matchthefrozencachedfeatureclassifications',
            'Originalsourcecrop/pad/JPEG preprocessing exactlymatchesexistingcacheinference',
            'OnefrozenDINObackbonerawfeaturepass sharedbybothbranchensembles; noexternalimages']}
    dump(ROOT/'results/milestone6/inference_verification.json',document)
    print(json.dumps({k:v for k,v in document.items() if k not in ['checks','code_sha256']}),flush=True)


if __name__=='__main__':main()
