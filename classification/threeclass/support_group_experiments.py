"""Fit-only rare-sibling marginalization and contrastive-positive controls."""
import fcntl
import itertools
import json
import os
from pathlib import Path
import traceback
import numpy as np
import torch
from experiment import ROOT,Model,load_data,normalize,train_one,summarize_group,metrics,dump
from kernel_residual_experiments import phase_plans,sha
from milestone2 import paired_oof

OUT=ROOT/'runs/support_group_taxonomy'


@torch.inference_mode()
def predict_cpu(c,features,indices):
    model=Model(c['cfg'],c['dims'],len(c['diseases'])).eval();model.load_state_dict(c['state_dict'],strict=True)
    probabilities=[]
    for begin in range(0,len(indices),256):
        selected=indices[begin:begin+256]
        x,_=normalize({key:value[selected] for key,value in features.items()},None,c['stats'])
        out=model({key:torch.from_numpy(value) for key,value in x.items()})
        original=out['subtype'].softmax(1)
        grouped=out['grouped_subtype_logits'].softmax(1)
        expected=torch.stack([original[:,model.support_groups.disease_to_group==g].sum(1)
            for g in range(model.support_groups.group_count)],1)
        assert torch.allclose(grouped,expected,atol=1e-6)
        probabilities.append(out['p'].numpy())
    return np.concatenate(probabilities)


def grouping_metadata(c):
    groups=c['state_dict']['support_groups.disease_to_group'].tolist()
    return [{'disease':name,'fit_patients':c['dims']['disease_patient_counts'][index],
        'family':c['dims']['disease_to_class'][index],'support_group':groups[index]}
        for index,name in enumerate(c['diseases'])]


def run():
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert json.loads((ROOT/'verification_support_group.json').read_text())['passed']
    reference=json.loads((ROOT/protocol['reference']).read_text())
    frame,features,digest=load_data('multiscale_vit');assert digest==protocol['manifest_sha256']
    records=[]
    for threshold,positive in itertools.product(protocol['thresholds'],protocol['contrastive_positive_modes']):
        name=f'support{threshold}_{positive}'
        cfg={**reference['selected']['config'],'support_group_threshold':threshold,'support_group_contrastive':positive=='support_group'}
        group=OUT/name/'validation'
        for seed in range(3):
            dump(OUT/'state.json',{'status':'formal_training','pid':os.getpid(),'arm':name,'seed':seed,'external_test_evaluated':False})
            train_one(frame,features,digest,{**cfg,'seed':seed},group/f'seed_{seed}')
        record=summarize_group(group,frame);record.update(name=name,candidate_id=name,threshold=threshold,contrastive_positives=positive)
        folds=[]
        for phase,fit,stop,outer,_ in phase_plans(frame,reference):
            evaluation=stop if phase=='validation' else outer
            fit_mask=np.zeros(len(frame),dtype=bool);stop_mask=fit_mask.copy();fit_mask[fit]=True;stop_mask[stop]=True
            probabilities=[];target=OUT/name/phase
            for seed in range(3):
                if phase!='validation':
                    dump(OUT/'state.json',{'status':'patient_cv_training','pid':os.getpid(),'arm':name,'phase':phase,'seed':seed,'external_test_evaluated':False})
                    train_one(frame,features,digest,{**cfg,'seed':seed},target/f'seed_{seed}',fit_mask,stop_mask)
                c=torch.load(target/f'seed_{seed}/model.pt',map_location='cpu',weights_only=False)
                p=predict_cpu(c,features,evaluation);probabilities.append(p)
                dump(target/f'seed_{seed}/support_groups.json',grouping_metadata(c))
            p=np.mean(probabilities,0)
            np.savez_compressed(target/'predictions.npz',indices=evaluation,p=p,pred=p.argmax(1))
            result={'phase':phase,'metrics':metrics(frame.iloc[evaluation].label,p)};dump(target/'metrics.json',result)
            if phase!='validation':
                np.savez_compressed(target/'outer_predictions.npz',indices=evaluation,p=p,pred=p.argmax(1));folds.append(result)
            else:assert abs(result['metrics']['macro_f1']-record['val']['macro_f1'])<1e-12
        record['cv']={'mean_macro_f1':float(np.mean([row['metrics']['macro_f1'] for row in folds])),
            'sd_macro_f1':float(np.std([row['metrics']['macro_f1'] for row in folds],ddof=1)),'folds':folds}
        record['paired_original_oof_vs_reference']=paired_oof(Path(reference['selected']['source_cv_summary']).parent,OUT/name,frame)
        record['development_gate_passed']=record['cv']['mean_macro_f1']>reference['cv']['mean_macro_f1'] and record['val']['macro_f1']>=reference['selected']['val']['macro_f1']
        records.append(record);dump(OUT/'leaderboard.json',records)
        print(json.dumps({'event':'support_group_arm_complete','arm':name,'val_f1':record['val']['macro_f1'],'cv_f1':record['cv']['mean_macro_f1']}),flush=True)
    destination=ROOT/'results/research_support_group'
    dump(destination/'comparison.json',{'manifest_sha256':digest,'protocol':protocol,'candidates':records,'new_seed_fits':72,'external_test_evaluated':False})
    audit_results(frame,features,digest,reference,records)
    eligible=[row for row in records if row['development_gate_passed']]
    if eligible:dump(destination/'selection_before_repeat.json',{'candidate':max(eligible,key=lambda row:row['cv']['mean_macro_f1']),'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'new_seed_fits':72,'passing_development_gates':len(eligible),'external_test_evaluated':False})
    from report import report
    report()


def audit_results(frame,features,digest,reference,records):
    checked=[];actual_groups=[]
    for record in records:
        scores=[];seen=[]
        for phase,fit,stop,outer,_ in phase_plans(frame,reference):
            evaluation=stop if phase=='validation' else outer
            for key in ['patient_group','split_group']:
                groups=[set(frame.iloc[ids][key]) for ids in [fit,stop,outer]]
                assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
            tf=frame.iloc[fit];counts=tf.groupby('disease').patient_group.nunique();mapping=tf.groupby('disease').label.first()
            probabilities=[]
            for seed in range(3):
                target=OUT/record['name']/phase/f'seed_{seed}'
                c=torch.load(target/'model.pt',map_location='cpu',weights_only=False)
                assert c['manifest_sha256']==digest and c['diseases']==sorted(tf.disease.unique())
                assert c['cfg']=={**reference['selected']['config'],'support_group_threshold':record['threshold'],
                    'support_group_contrastive':record['contrastive_positives']=='support_group','seed':seed}
                assert c['dims']['disease_patient_counts']==[int(counts[name]) for name in c['diseases']]
                assert c['dims']['disease_to_class']==[int(mapping[name]) for name in c['diseases']]
                groups=c['state_dict']['support_groups.disease_to_group'].numpy()
                for i,a in enumerate(c['diseases']):
                    for j,b in enumerate(c['diseases']):
                        expected=(i==j) or (mapping[a]==mapping[b] and counts[a]<record['threshold'] and counts[b]<record['threshold'])
                        assert (groups[i]==groups[j])==expected
                for key,values in features.items():
                    z=values[fit];axes=tuple(range(z.ndim-1))
                    assert np.allclose(c['stats'][key]['mean'],z.mean(axes),atol=1e-6,rtol=1e-6)
                    assert np.allclose(c['stats'][key]['std'],np.maximum(z.std(axes),.05),atol=1e-6,rtol=1e-6)
                stop_p=predict_cpu(c,features,stop);saved=np.load(target/'val_predictions.npz')['p']
                assert np.allclose(stop_p,saved,atol=5e-5,rtol=5e-5) and np.array_equal(stop_p.argmax(1),saved.argmax(1))
                probabilities.append(stop_p if phase=='validation' else predict_cpu(c,features,outer))
                stored=json.loads((target/'support_groups.json').read_text());assert stored==grouping_metadata(c)
                actual_groups.append({'arm':record['name'],'phase':phase,'seed':seed,'groups':stored})
                checked.append({'arm':record['name'],'phase':phase,'seed':seed})
            p=np.mean(probabilities,0);saved=np.load(OUT/record['name']/phase/'predictions.npz')
            assert np.array_equal(saved['indices'],evaluation) and np.array_equal(saved['pred'],p.argmax(1))
            assert np.allclose(saved['p'],p,atol=5e-5,rtol=5e-5)
            actual=metrics(frame.iloc[evaluation].label,saved['p']);stored=json.loads((OUT/record['name']/phase/'metrics.json').read_text())['metrics']
            for key in ['accuracy','macro_f1','log_loss']:assert abs(actual[key]-stored[key])<1e-12
            if phase!='validation':scores.append(actual['macro_f1']);seen.extend(evaluation.tolist())
        assert sorted(seen)==np.flatnonzero(frame.split.eq('train')).tolist()
        assert abs(np.mean(scores)-record['cv']['mean_macro_f1'])<1e-12
        print(json.dumps({'event':'support_group_arm_cpu_audited','arm':record['name'],'actual_heads':12}),flush=True)
    assert len(checked)==72
    destination=ROOT/'results/research_support_group'
    dump(destination/'actual_groups.json',{'models':actual_groups})
    dump(destination/'verification.json',{'passed':True,'manifest_sha256':digest,'actual_seed_heads_checked':72,'fits':checked,
        'checks':['All6fixedauxiliarygrouping/contrastivearmscompletedall12actualseedheads',
            'Every patient/duplicate split disjoint; fit-only feature statistics and original disease maps reconstructed',
            'Every disease support count independently reconstructed from current fitting patients',
            'All pairwise disease grouping relationships match same-family support threshold rule',
            'Every actual stopping and outer prediction independently recovered onCPU',
            'Grouped probability mass equals summed original disease mass for all checked batches',
            'All ensemble metrics, class decisions and original outer image coverage reconstructed']})


def main():
    torch.set_num_threads(4)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(OUT/'runner.pid').write_text(str(os.getpid()))
        dump(OUT/'implementation.json',{'code_sha256':{name:sha(ROOT/name) for name in ['support_group_experiments.py','support_group.py','experiment.py']}})
        try:run()
        except BaseException:
            dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc()});raise


if __name__=='__main__':main()
