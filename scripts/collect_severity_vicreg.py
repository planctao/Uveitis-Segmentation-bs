"""Verify complete five-fold VICReg budgets, encoder reload and all scores."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bs.severity_vicreg import ImageVICReg
from bs.severity_synthetic import SYNTHETIC_METHODS


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);args=parser.parse_args()
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25)
    root=ROOT/'outputs/severity_research_20261013'
    images=pd.read_csv(root/'data/image_groups.csv',dtype={'dhash':str}).set_index('uid')
    out=root/'domain_ssl/vicreg_evaluation'/f'seed{args.seed}';out.mkdir(parents=True,exist_ok=True)
    predictions=[];audits=[];splits=[];sources={}
    for fold in ['f1','f2','f3','f4','f5']:
        run=root/'domain_ssl/vicreg'/f'seed{args.seed}'/fold
        complete=json.loads((run/'complete.json').read_text());plan=json.loads((run/'plan.json').read_text())
        for name,expected in complete['source_sha256'].items():
            assert digest(ROOT/name)==expected,name;sources[name]=expected
        path=run/'predictions_frozen_before_evaluation.csv'
        assert digest(path)==complete['predictions_sha256']
        pred=pd.read_csv(path)
        train=images.loc[plan['train_uids']];test=images.loc[plan['test_uids']]
        assert not train.exclude_exact_group.any() and not test.exclude_exact_group.any()
        assert not set(train.pixel_sha256)&set(test.pixel_sha256)
        assert (train.fold!=fold).all() and (test.fold==fold).all()
        assert set(train.index)|set(test.index)==set(images.index[~images.exclude_exact_group])
        trace=pd.read_csv(run/'training_trace.csv')
        assert complete['epochs']==20 and len(trace)==20 and trace.epoch.tolist()==list(range(1,21))
        assert (trace.n_seen==(len(train)//16)*16).all()
        assert (trace.successful_updates+trace.amp_skips==len(train)//16).all()
        assert int(trace.successful_updates.sum())==complete['optimizer_updates']
        assert int(trace.amp_skips.sum())==complete['amp_skips']
        assert np.isfinite(trace[['mean_loss','mean_invariance','mean_variance','mean_covariance']].to_numpy()).all()
        assert trace.encoder_stem_max_abs_update.iloc[-1]>0
        assert len(pred)==len(test)*4*3
        for _,g in pred.groupby(['method','gamma']):assert set(g.uid)==set(test.index)
        rep=np.load(run/'final_representations.npz',allow_pickle=False)
        assert np.array_equal(rep['uids'],images.index.to_numpy())
        uid_index={uid:k for k,uid in enumerate(rep['uids'])}
        test_indices=np.array([uid_index[u] for u in test.index])
        checkpoint=torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
        model=ImageVICReg(ROOT/'weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth').cuda().eval().requires_grad_(False)
        model.load_state_dict(checkpoint['model'],strict=True)
        mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1)
        std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1)
        neural_error=0.;projection_error=0.
        for offset in sorted(set((test_indices[:3]//8*8).tolist())):
            batch_uids=rep['uids'][offset:offset+8]
            rgb=np.stack([np.asarray(Image.open(images.loc[u,'image_path']).convert('RGB').resize((256,256),Image.Resampling.BILINEAR)) for u in batch_uids])
            tensor=torch.from_numpy(rgb).permute(0,3,1,2).float().cuda()/255
            with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.float16):
                encoded,projected=model((tensor-mean)/std)
            neural_error=max(neural_error,float(np.max(np.abs(encoded.float().cpu().numpy()-rep['features'][0,offset:offset+8]))))
            projection_error=max(projection_error,float(np.max(np.abs(projected.float().cpu().numpy()-rep['projections'][0,offset:offset+8]))))
        assert neural_error<.02 and projection_error<.02
        del model;torch.cuda.empty_cache()
        score_error=0.
        for method in SYNTHETIC_METHODS:
            calibration=joblib.load(run/f'vicreg_encoder_{method}.joblib')
            assert calibration.diagnostics['n_fit']==len(train)
            for gi,gamma in enumerate([1.,.8,1.2]):
                score,level=calibration.predict(rep['features'][gi,test_indices])
                frozen=pred[(pred.method==f'vicreg_encoder_{method}')&(pred.gamma==gamma)].set_index('uid').loc[test.index]
                difference=float(np.max(np.abs(score-frozen.score)));score_error=max(score_error,difference)
                assert difference<1e-12 and np.array_equal(level,frozen.level)
        predictions.append(pred)
        splits.append(dict(scope='main',fold=fold,n_train=len(train),n_test=len(test),train_uids=list(train.index),test_uids=list(test.index)))
        audits.append(dict(fold=fold,epochs=20,budget_verified=True,actual_neural_encoder_reload_verified=True,
            neural_feature_max_error=neural_error,neural_projection_max_error=projection_error,
            all_heldout_calibration_scores_replayed=True,max_score_error=score_error,exact_hash_overlap=0,
            optimizer_updates=complete['optimizer_updates'],amp_skips=complete['amp_skips']))
    predictions=pd.concat(predictions,ignore_index=True)
    assert predictions.uid.nunique()==2055 and len(predictions)==24660
    assert not predictions.duplicated(['uid','method','gamma']).any()
    path=out/'predictions_frozen_before_evaluation.csv';predictions.to_csv(path,index=False)
    sources[str(Path(__file__).relative_to(ROOT))]=digest(Path(__file__))
    (out/'splits.json').write_text(json.dumps(splits,indent=2)+'\n')
    (out/'frozen_manifest.json').write_text(json.dumps(dict(seed=args.seed,family='vicreg',source_sha256=sources,
        predictions_sha256=digest(path),no_mask_or_clinical_label_inputs=True,five_fold_budget_verified=True,
        clinical_grading_validated=False,patient_independence_proven=False,audits=audits),indent=2)+'\n')
    print('VICReg five-fold audit complete',args.seed,len(predictions),flush=True)


if __name__=='__main__':main()
