"""Audit complete five-fold states, then freeze a seed's evaluation input."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bs.severity_ssl import DECEncoder
from bs.severity_swav_images import ImageSwAV


def main():
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['dec','swav'],required=True);p.add_argument('--seed',type=int,required=True)
    args=p.parse_args();torch.set_num_threads(4)
    root=ROOT/'outputs/severity_research_20261013'
    images=pd.read_csv(root/'data/image_groups.csv',dtype={'dhash':str}).set_index('uid')
    out=root/'domain_ssl'/f'{args.family}_evaluation'/f'seed{args.seed}';out.mkdir(parents=True,exist_ok=True)
    predictions=[];sources={};audit=[];splits=[]
    original=np.load(root/'embeddings/frozen_dino_gamma.npz',allow_pickle=False)
    uid_index={uid:i for i,uid in enumerate(original['uids'])}
    for fold in ['f1','f2','f3','f4','f5']:
        run=root/'domain_ssl'/args.family/f'seed{args.seed}'/fold
        complete=json.loads((run/'complete.json').read_text());plan=json.loads((run/'plan.json').read_text())
        for name,digest in complete['source_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
            sources[name]=digest
        prediction_path=run/'predictions_frozen_before_evaluation.csv'
        assert hashlib.sha256(prediction_path.read_bytes()).hexdigest()==complete['predictions_sha256']
        pred=pd.read_csv(prediction_path);predictions.append(pred)
        train=images.loc[plan['train_uids']];test=images.loc[plan['test_uids']]
        assert not train.exclude_exact_group.any() and not test.exclude_exact_group.any()
        assert not set(train.pixel_sha256)&set(test.pixel_sha256)
        for _,g in pred.groupby(['method','gamma']):assert set(g.uid)==set(test.index)
        trace=pd.read_csv(run/'training_trace.csv')
        if args.family=='dec':
            assert complete['phase_epochs']=={'ae_control':200,'dec':200,'pretrain':200}
            assert complete['optimizer_updates']==600*((len(train)+127)//128)
            assert complete['shared_initial_tensors_match']
            # Recompute actual held-out representations from each neural encoder.
            for branch in ['ae','dec']:
                state=torch.load(run/f'{branch}_final.pt',map_location='cpu',weights_only=False)
                model=DECEncoder().eval();model.load_state_dict(state['model'],strict=True)
                representations=np.load(run/f'{branch}_representations.npz',allow_pickle=False)
                sample=test.index[:3];idx=np.array([uid_index[u] for u in sample])
                raw=original['embeddings'][0,idx];norm=raw/np.linalg.norm(raw,axis=1,keepdims=True)
                inp=state['scaler'].transform(norm).astype(np.float32)
                with torch.inference_mode():computed=model.encoder(torch.from_numpy(inp)).numpy()
                diff=float(np.max(np.abs(computed-representations['features'][0,idx])))
                assert diff<2e-4
                for method in ['synthetic_direction','gamma_orthogonal_direction']:
                    calibration=joblib.load(run/f'{branch}_{method}.joblib')
                    score,level=calibration.predict(representations['features'][0,idx])
                    saved=pred[(pred.method==f'{branch}_{method}')&(pred.gamma==1.)].set_index('uid').loc[sample]
                    assert np.allclose(score,saved.score,atol=1e-12) and np.array_equal(level,saved.level)
        else:
            assert complete['epochs']==20 and len(trace)==20
            assert (trace.successful_updates+trace.amp_skips).sum()==20*(len(train)//8)
            assert trace.encoder_stem_max_abs_update.iloc[-1]>0
            representations=np.load(run/'final_representations.npz',allow_pickle=False)
            state=torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
            model=ImageSwAV(ROOT/'weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth').cuda().eval().requires_grad_(False)
            model.load_state_dict(state['model'],strict=True)
            sample=test.index[:3];sample_indices=np.array([uid_index[u] for u in sample])
            mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1)
            std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1)
            neural_max_difference=0.
            for offset in sorted(set((sample_indices//8*8).tolist())):
                batch_uids=original['uids'][offset:offset+8]
                rgb=np.stack([np.asarray(Image.open(images.loc[u,'image_path']).convert('RGB').resize((256,256),Image.Resampling.BILINEAR)) for u in batch_uids])
                tensor=torch.from_numpy(rgb).permute(0,3,1,2).float().cuda()/255
                with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.float16):
                    recomputed,_,_=model((tensor-mean)/std)
                difference=float(np.max(np.abs(recomputed.float().cpu().numpy()-representations['features'][0,offset:offset+8])))
                neural_max_difference=max(neural_max_difference,difference)
            assert neural_max_difference<.02
            del model;torch.cuda.empty_cache()
            for method in ['synthetic_direction','meanmatched_direction','gamma_orthogonal_direction','synthetic_oriented_pc1']:
                sample=test.index[:3];idx=np.array([uid_index[u] for u in sample])
                calibration=joblib.load(run/f'swav_encoder_{method}.joblib')
                score,level=calibration.predict(representations['features'][0,idx])
                saved=pred[(pred.method==f'swav_encoder_{method}')&(pred.gamma==1.)].set_index('uid').loc[sample]
                assert np.allclose(score,saved.score,atol=1e-12) and np.array_equal(level,saved.level)
        audit.append(dict(fold=fold,epochs_verified=True,source_hashes_match=True,prediction_hash_matches=True,
                          exact_duplicate_overlap=0,n_fit=len(train),n_test=len(test),updates=complete['optimizer_updates'],
                          calibration_reload_verified=True,neural_encoder_reload_verified=True))
        splits.append(dict(scope='main',fold=fold,train_uids=list(train.index),test_uids=list(test.index),n_train=len(train),n_test=len(test)))
    all_pred=pd.concat(predictions,ignore_index=True)
    assert all_pred.uid.nunique()==2055 and not all_pred.duplicated(['uid','scope','fold','gamma','method']).any()
    path=out/'predictions_frozen_before_evaluation.csv';all_pred.to_csv(path,index=False)
    sources[str(Path(__file__).relative_to(ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'splits.json').write_text(json.dumps(splits,indent=2)+'\n')
    (out/'frozen_manifest.json').write_text(json.dumps(dict(seed=args.seed,family=args.family,source_sha256=sources,
        predictions_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),no_mask_or_clinical_label_inputs=True,
        five_fold_budget_verified=True,clinical_grading_validated=False,patient_independence_proven=False,
        audits=audit),indent=2)+'\n')
    print(args.family,args.seed,'complete five-fold artifact audit passed',len(all_pred),'predictions',flush=True)


if __name__=='__main__':main()
