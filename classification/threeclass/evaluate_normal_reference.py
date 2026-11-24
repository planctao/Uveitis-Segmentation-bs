"""Explicit one-time external evaluation after development, fresh-patient and inference review."""
import fcntl
import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch
from experiment import ROOT,CLASSES,load_data,metrics,dump
from evaluate import cluster_accuracy_ci,aggregate
from predict_normal_reference import FrozenNormalReferencePredictor
from kernel_residual_experiments import sha


def paired_accuracy(base,new):
    a=pd.read_csv(base);b=pd.read_csv(new)
    assert np.array_equal(a.path,b.path) and np.array_equal(a.label,b.label)
    assert np.array_equal(a.patient_group,b.patient_group)
    _,group=np.unique(a.patient_group,return_inverse=True)
    count=np.bincount(group)
    delta=np.bincount(group,weights=(b.prediction==b.label).astype(float)-(a.prediction==a.label).astype(float))
    samples=np.random.default_rng(20261124).integers(0,len(count),size=(2000,len(count)))
    differences=delta[samples].sum(1)/count[samples].sum(1)
    return {'accuracy_difference':float(delta.sum()/count.sum()),'patient_cluster_bootstrap_ci95':np.quantile(differences,[.025,.975]).tolist(),'bootstrap_repetitions':2000}


def run():
    out=ROOT/'results/milestone6';selection_path=out/'selection_before_test.json'
    selection=json.loads(selection_path.read_text());selection_hash=sha(selection_path)
    for name,expected in selection['evaluation_code_sha256'].items():assert sha(ROOT/name)==expected
    assert not (out/'metrics.json').exists(),'This frozen milestone already has its external evaluation.'
    predictor=FrozenNormalReferencePredictor(device='cuda')
    assert predictor.decision['candidate_id']==selection['candidate_id']
    assert sha(ROOT/'features/multiscale_vit.npz')==predictor.decision['feature_bank_sha256']
    frame,features,digest=load_data('multiscale_vit');assert digest==predictor.decision['manifest_sha256']
    dump(out/'evaluation_started.json',{'time_utc':datetime.now(timezone.utc).isoformat(),'selection_sha256':selection_hash,
        'policy':'Frozen single candidate; evaluate once, without selection or parameter changes.'})
    results={};saved_probabilities={}
    for split in ['val','test_independent','test_ningbo']:
        indices=np.flatnonzero(frame.split.eq(split));df=frame.iloc[indices].reset_index(drop=True);parts=[]
        for begin in range(0,len(indices),128):
            chosen=indices[begin:begin+128]
            parts.append(predictor.predict_features({key:value[chosen] for key,value in features.items()}))
        p=np.concatenate(parts);pred=p.argmax(1);saved_probabilities[split]=p
        np.savez_compressed(out/f'{split}_predictions.npz',indices=indices,p=p,pred=pred)
        result={'image':metrics(df.label,p),'image_accuracy_patient_cluster_ci95':cluster_accuracy_ci(df,pred)}
        image=df[['path','cache_path','patient_group','eye_group','disease','label']].copy()
        for i,name in enumerate(CLASSES):image['p_'+name]=p[:,i]
        image['prediction']=pred;image.to_csv(out/f'{split}_images.csv',index=False)
        for key in ['patient_group','eye_group']:
            aggregated,ap,_,excluded=aggregate(df,p,np.zeros((len(p),0),dtype='float32'),key)
            name='patient' if key=='patient_group' else 'eye'
            result[name]=metrics(aggregated.label,ap);result[name]['excluded_mixed_label_groups']=excluded
            for i,cls in enumerate(CLASSES):aggregated['p_'+cls]=ap[:,i]
            aggregated['prediction']=ap.argmax(1);aggregated.to_csv(out/f'{split}_{name}.csv',index=False)
        pd.DataFrame([{'disease':name,'n':len(group),'accuracy':float((group.label==group.prediction).mean())} for name,group in image.groupby('disease')]).to_csv(out/f'{split}_by_disease.csv',index=False)
        results[split]=result
    # Independently reload all six checkpoints on CPU and recover every evaluated row.
    cpu=FrozenNormalReferencePredictor(device='cpu');errors={}
    for split,p in saved_probabilities.items():
        indices=np.flatnonzero(frame.split.eq(split));parts=[]
        for begin in range(0,len(indices),128):
            chosen=indices[begin:begin+128]
            parts.append(cpu.predict_features({key:value[chosen] for key,value in features.items()}))
        independent=np.concatenate(parts)
        assert np.allclose(p,independent,atol=5e-5,rtol=5e-5) and np.array_equal(p.argmax(1),independent.argmax(1))
        table=pd.read_csv(out/f'{split}_images.csv')
        assert np.array_equal(table.path,frame.iloc[indices].path) and np.array_equal(table.prediction,independent.argmax(1))
        actual=metrics(table.label,p)
        for key in ['accuracy','macro_f1','log_loss']:assert abs(actual[key]-results[split]['image'][key])<1e-12
        errors[split]=float(np.max(np.abs(p-independent)))
    assert sha(selection_path)==selection_hash and sha(out/'package.json')==selection['package_sha256']
    for name,expected in selection['evaluation_code_sha256'].items():assert sha(ROOT/name)==expected
    paired={}
    for split in ['test_independent','test_ningbo']:
        paired[split]={'versus_milestone5':paired_accuracy(ROOT/'results/milestone5/primary'/f'{split}_images.csv',out/f'{split}_images.csv'),
            'versus_fixed_gaussian':paired_accuracy(ROOT/'results/milestone_gaussian'/f'{split}_images.csv',out/f'{split}_images.csv')}
    dump(out/'paired_external_comparison.json',paired)
    dump(out/'metrics.json',{'selection':selection,'manifest_sha256':digest,'class_order':CLASSES,'results':results})
    dump(out/'verification.json',{'passed':True,'time_utc':datetime.now(timezone.utc).isoformat(),
        'selection_sha256':selection_hash,'package_sha256':selection['package_sha256'],
        'all_image_predictions_cpu_restored':True,'maximum_cpu_probability_errors':errors,
        'checks':['Developmentandfreshpatientgatesandinferencereview passedbeforeexternalfreeze',
            'Allsiximmutablemodelcheckpoints,normalizationandfixed0.75mixture verified',
            'Everyvalidationandexternalscore independentlyrecoveredonCPU; classdecisionsandCSVsidentical',
            'Manifestrows/confusionmatrices,patient/eyeaggregationandpairedpatientaccuracycomparisonschecked',
            'Finalselection,modelpackageandevaluationcodeunchanged; noexternalparameterselection']})
    print(json.dumps({'candidate_id':selection['candidate_id'],'external':{s:v['image'] for s,v in results.items() if s.startswith('test')},'paired':paired}),flush=True)


def main():
    torch.set_num_threads(4)
    with (ROOT/'results/milestone6/evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run()


if __name__=='__main__':main()
