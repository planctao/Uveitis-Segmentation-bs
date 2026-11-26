"""Descriptive subgroup and sign diagnostics of already frozen predictions."""
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bs.dataset import decode_mask_array
from bs.severity_ordering import rho
from evaluate_severity_baselines import bootstrap_correlations


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_rho(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return float(rho(a,b)) if len(a)>=3 and np.ptp(a)>0 and np.ptp(b)>0 else np.nan


def partial_rank_correlation(score,target,controls):
    y=rankdata(target)
    x=rankdata(score)
    c=np.column_stack([np.ones(len(x)),*[rankdata(v) for v in controls]])
    x-=c@np.linalg.lstsq(c,x,rcond=None)[0]
    y-=c@np.linalg.lstsq(c,y,rcond=None)[0]
    return float(np.corrcoef(x,y)[0,1]) if min(np.std(x),np.std(y))>1e-12 else np.nan


def main():
    root=ROOT/'outputs/severity_research_20261013'
    out=root/'failure_audit'
    out.mkdir(exist_ok=True)
    models=[
        ('joint_gamma',root/'registration_stress/seed42','gamma_orthogonal_direction','registration_group_stress'),
        ('synthetic_pc1',root/'registration_stress/seed42','synthetic_oriented_pc1','registration_group_stress'),
        ('mean_brightness',root/'baselines/seed42','mean','main'),
        ('ae_gamma',root/'domain_ssl/dec_evaluation/seed42','ae_gamma_orthogonal_direction','main')]
    inputs,score_frames={},[]
    for name,run,method,scope in models:
        manifest=json.loads((run/'frozen_manifest.json').read_text())
        path=run/'predictions_frozen_before_evaluation.csv'
        assert digest(path)==manifest['predictions_sha256']
        for source,expected in manifest['source_sha256'].items():assert digest(ROOT/source)==expected,source
        inputs[str(path.relative_to(ROOT))]=digest(path)
        pred=pd.read_csv(path)
        pred=pred[(pred.gamma==1.)&(pred.method==method)&(pred.scope==scope)].copy()
        assert len(pred)==2055 and pred.uid.nunique()==2055
        score_frames.append(pred.assign(model=name,training_scope=scope))
    predictions=pd.concat(score_frames,ignore_index=True)
    images=pd.read_csv(root/'data/image_groups.csv',dtype={'dhash':str}).set_index('uid')
    features=np.load(root/'embeddings/frozen_dino_gamma.npz',allow_pickle=False)
    descriptors=pd.DataFrame(dict(uid=features['uids'],brightness=features['image_features'][0,:,0],fov_fraction=features['image_features'][0,:,6])).set_index('uid')
    images=images.join(descriptors,validate='one_to_one')
    splits=json.loads((root/'registration_stress/seed42/splits.json').read_text())
    strata=[]
    for split in splits:
        train=images.loc[split['train_uids']]
        test=images.loc[split['test_uids']]
        for column in ['brightness','fov_fraction']:
            cuts=np.quantile(train[column],[.25,.5,.75])
            images.loc[test.index,f'{column}_quartile']=np.searchsorted(cuts,test[column],side='right')+1
            strata.append(dict(fold=split['fold'],descriptor=column,n_fit=len(train),cutpoints=cuts.tolist()))
    # Open evaluation targets only after all four frozen prediction audits pass.
    target_path=root/'data/mask_burden_evaluation_only.csv'
    targets=pd.read_csv(target_path).set_index('uid')
    data=predictions.merge(images.reset_index(),on=['uid','fold'],validate='many_to_one').merge(targets.reset_index(),on='uid',validate='many_to_one')
    data['lesion2_present']=data.lesion2_fraction>0
    groups=[('all','all',np.ones(len(data),dtype=bool))]
    for column in ['brightness_quartile','fov_fraction_quartile']:
        for q in range(1,5):groups.append((column,str(q),data[column].to_numpy()==q))
    for value in [False,True]:groups.append(('lesion2_present',str(value),data.lesion2_present.to_numpy()==value))
    for value in sorted(data.extension.unique()):groups.append(('file_format',value,data.extension.to_numpy()==value))
    metrics=[]
    for stratum,value,selector in groups:
        selected=data[selector]
        for model,_,_,_ in models:
            group=selected[selected.model==model].sort_values('uid')
            n=len(group)
            union_rho=safe_rho(group.score,group.union_fraction)
            ci=[np.nan,np.nan]
            if n>=20 and np.isfinite(union_rho):
                ci=np.quantile(bootstrap_correlations(group.score.to_numpy()[None,:],group.union_fraction.to_numpy())[0],[.025,.975])
            auc=float(roc_auc_score(group.lesion2_present,group.score)) if group.lesion2_present.nunique()==2 else np.nan
            metrics.append(dict(stratum=stratum,value=value,model=model,n=n,n_lesion2_positive=int(group.lesion2_present.sum()),
                union_rho=union_rho,ci_low=float(ci[0]),ci_high=float(ci[1]),
                lesion1_rho=safe_rho(group.score,group.lesion1_fraction),lesion2_rho=safe_rho(group.score,group.lesion2_fraction),
                lesion2_presence_auc=auc,median_union_fraction=float(group.union_fraction.median()) if n else np.nan))
    table=pd.DataFrame(metrics)
    table.to_csv(out/'subgroup_metrics.csv',index=False)
    partial=[]
    for model,g in data.groupby('model'):
        positive=g[g.lesion2_present]
        partial.append(dict(model=model,n=len(g),n_lesion2_positive=len(positive),
            lesion2_positive_only_area_rho=safe_rho(positive.score,positive.lesion2_fraction),
            partial_lesion2_rho_given_lesion1=partial_rank_correlation(g.score,g.lesion2_fraction,[g.lesion1_fraction]),
            partial_lesion2_rho_given_lesion1_and_brightness=partial_rank_correlation(g.score,g.lesion2_fraction,[g.lesion1_fraction,g.brightness])))
    pd.DataFrame(partial).to_csv(out/'conditional_sign_metrics.csv',index=False)
    pairs=[]
    for (model,fold),g in data.groupby(['model','fold']):
        score,area=g.score.to_numpy(),g.union_fraction.to_numpy()
        a,b=np.triu_indices(len(g),k=1)
        area_difference=area[a]-area[b]
        score_difference=score[a]-score[b]
        direction=score_difference*np.sign(area_difference)
        for threshold in [.002,.01,.05]:
            valid=np.abs(area_difference)>=threshold
            n=int(valid.sum())
            correct=int((direction[valid]>0).sum())
            ties=int((direction[valid]==0).sum())
            pairs.append(dict(model=model,fold=fold,minimum_area_difference=threshold,n_pairs=n,
                n_correct=correct,n_score_ties=ties,pair_correct_fraction=(correct+.5*ties)/n if n else np.nan))
    pair_table=pd.DataFrame(pairs)
    pair_table.to_csv(out/'within_fold_area_pairs.csv',index=False)
    pooled=pair_table.groupby(['model','minimum_area_difference'])[['n_pairs','n_correct','n_score_ties']].sum().reset_index()
    pooled['pair_correct_fraction']=(pooled.n_correct+.5*pooled.n_score_ties)/pooled.n_pairs
    pooled.to_csv(out/'pooled_area_pair_metrics.csv',index=False)
    primary=data[data.model=='joint_gamma'].copy()
    primary['heldout_area_percentile']=primary.groupby('fold').union_fraction.transform(lambda a:(rankdata(a)-.5)/len(a))
    primary['rank_residual']=primary.score-primary.heldout_area_percentile
    primary.to_csv(out/'all_primary_rank_residuals.csv',index=False)
    over=primary.sort_values(['rank_residual','uid'],ascending=[False,True]).head(6).copy()
    under=primary.sort_values(['rank_residual','uid'],ascending=[True,True]).head(6).copy()
    over['example_id']=[f'O{k+1}' for k in range(6)]
    under['example_id']=[f'U{k+1}' for k in range(6)]
    selected=pd.concat([over.assign(selection='higher_score_than_area_rank'),under.assign(selection='lower_score_than_area_rank')])
    selected.to_csv(out/'selected_posthoc_examples.csv',index=False)
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(font_path,13) if Path(font_path).exists() else ImageFont.load_default()
    montage=Image.new('RGB',(1024,6*316+32),'#101010')
    draw=ImageDraw.Draw(montage)
    draw.text((5,5),'Post hoc area-rank differences; original + archived mask; O higher / U lower',fill='white',font=font)
    mask_audits=[]
    palette=np.array([[0,0,0],[255,64,64],[64,210,110],[80,150,255]],dtype=np.uint8)
    for side,group in enumerate([over,under]):
        for row,(idx,r) in enumerate(group.iterrows()):
            image_path=Path(r.image_path)
            mask_path=image_path.parents[2]/'mask_only_itksnap'/r.fold/(image_path.stem+'.png')
            rgb=Image.open(image_path).convert('RGB')
            pixel_hash=hashlib.sha256(np.asarray(rgb).tobytes()+str(rgb.size).encode()).hexdigest()
            assert pixel_hash==r.pixel_sha256
            mask=decode_mask_array(np.asarray(Image.open(mask_path)),mask_path)
            mask_hash=hashlib.sha256(mask.tobytes()).hexdigest()
            assert mask_hash==r.mask_sha256
            assert abs((mask>0).mean()-r.union_fraction)<1e-12
            small=rgb.resize((256,256),Image.Resampling.BILINEAR)
            labels=np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((256,256),Image.Resampling.NEAREST))
            overlay=np.asarray(small).copy()
            valid=labels>0
            overlay[valid]=np.rint(.5*overlay[valid]+.5*palette[labels[valid]]).astype(np.uint8)
            x,y=side*512,32+row*316
            montage.paste(small,(x,y));montage.paste(Image.fromarray(overlay),(x+256,y))
            draw.text((x+3,y+258),f'{r.example_id} {r.uid[:7]} {r.fold} score={r.score:.3f} area={r.union_fraction*100:.2f}%',fill='white',font=font)
            draw.text((x+3,y+278),f'area rank={r.heldout_area_percentile:.3f}; residual={r.rank_residual:+.3f}; lesion2={r.lesion2_fraction*100:.2f}%',fill='white',font=font)
            mask_audits.append(dict(uid=r.uid,decoded_mask_sha256=mask_hash,pixel_sha256=pixel_hash))
    montage.save(out/'posthoc_area_rank_examples.jpg',quality=95)
    fig,axs=plt.subplots(1,2,figsize=(11,4.7),constrained_layout=True)
    for model,color in [('joint_gamma','#31688e'),('mean_brightness','#b26c35')]:
        values=table[(table.stratum=='brightness_quartile')&(table.model==model)].sort_values('value')
        axs[0].errorbar(np.arange(1,5),values.union_rho,yerr=[values.union_rho-values.ci_low,values.ci_high-values.union_rho],fmt='o-',capsize=3,label=model,color=color)
    axs[0].set_xticks(range(1,5));axs[0].set_xlabel('Brightness quartile defined on fit-fold images')
    axs[0].set_ylabel('Held-out union area Spearman rho');axs[0].legend()
    axs[0].set_title('Descriptive subgroups; image bootstrap 95% CI')
    axs[1].scatter(primary.union_fraction*100,primary.score,s=7,alpha=.2,color='#555555')
    for frame,color in [(over,'#bd5151'),(under,'#337ab7')]:
        axs[1].scatter(frame.union_fraction*100,frame.score,s=28,color=color)
        for r in frame.itertuples():axs[1].annotate(r.example_id,(r.union_fraction*100,r.score),fontsize=7)
    axs[1].set_xlabel('Archived union mask area (% of full image)')
    axs[1].set_ylabel('Frozen score percentile');axs[1].set_title('Selected differences are post hoc area proxies')
    fig.savefig(out/'subgroups_and_area_rank.png',dpi=180);fig.savefig(out/'subgroups_and_area_rank.pdf');plt.close(fig)
    protocol=ROOT/'docs/severity_failure_audit_plan_20261013.md'
    summary=dict(n_images=2055,models=[dict(name=a,method=c,training_scope=d) for a,_,c,d in models],
        prediction_inputs_sha256=inputs,target_sha256=digest(target_path),
        image_features_sha256=digest(root/'embeddings/frozen_dino_gamma.npz'),
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),protocol]},
        fit_fold_strata=strata,selected_example_masks_and_pixels=mask_audits,
        posthoc_analysis=True,model_refitted_or_selected=False,clinical_grading_validated=False,patient_independence_proven=False,
        pair_counts_are_dependent=True,n_bootstrap=1000,confidence_interval_unit='image')
    (out/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(table[(table.stratum=='all')|((table.stratum=='brightness_quartile')&(table.model=='joint_gamma'))].to_string(index=False),flush=True)
    print(pd.DataFrame(partial).to_string(index=False),flush=True)
    print(pooled.to_string(index=False),flush=True)


if __name__=='__main__':main()
