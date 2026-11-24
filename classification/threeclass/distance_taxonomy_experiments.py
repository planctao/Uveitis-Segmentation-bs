"""Fixed Euclidean/Poincare disease prototypes, with complete independent CPU audit."""
import fcntl
import json
import os
from pathlib import Path
import traceback
import numpy as np
import torch
from experiment import ROOT,load_data,normalize,train_one,summarize_group,metrics,dump
from kernel_residual_experiments import phase_plans,sha
from milestone2 import paired_oof
from distance_taxonomy import DistanceTaxonomyModel
from verify_distance_taxonomy import independent_distance

OUT=ROOT/'runs/distance_taxonomy'
DEST=ROOT/'results/research_distance_taxonomy'


def configuration(reference,arm):
    return {**reference['selected']['config'],'distance_rank':arm['rank'],'distance_geometry':arm['geometry']}


@torch.inference_mode()
def predict_cpu(c,features,indices,diagnostics=False):
    assert len(indices) and np.max(indices)<2091
    model=DistanceTaxonomyModel(c['cfg'],c['dims'],len(c['diseases'])).eval();model.load_state_dict(c['state_dict'],strict=True)
    ps=[];norms=[];distances=[];errors=[]
    for begin in range(0,len(indices),256):
        ids=indices[begin:begin+256];x,_=normalize({k:v[ids] for k,v in features.items()},None,c['stats'])
        assert all(np.isfinite(v).all() for v in x.values())
        out=model({k:torch.from_numpy(v) for k,v in x.items()});ps.append(out['p'].numpy())
        if diagnostics:
            expected,embedding,centers,distance=independent_distance(model.flat,out['embedding'].numpy())
            assert np.allclose(expected,out['subtype'].numpy(),atol=5e-5,rtol=5e-5)
            norms.append(np.linalg.norm(embedding,axis=1));distances.append(distance)
            errors.append(float(np.max(np.abs(expected-out['subtype'].numpy()))))
            if c['cfg']['distance_geometry']=='poincare':
                assert np.max(norms[-1])<1 and np.max(np.linalg.norm(centers,axis=1))<1
    if not diagnostics:return np.concatenate(ps)
    return np.concatenate(ps),{'mean_embedding_norm':float(np.concatenate(norms).mean()),
        'max_embedding_norm':float(np.concatenate(norms).max()),
        'mean_squared_distance':float(np.concatenate(distances).mean()),
        'maximum_logit_formula_error':max(errors),'inverse_temperature':float(model.flat.log_scale.exp().clamp(1,100))}


def run():
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert json.loads((ROOT/'verification_distance_taxonomy.json').read_text())['passed']
    for name,digest in protocol['code_sha256'].items():assert sha(ROOT/name)==digest
    reference=json.loads((ROOT/protocol['reference']).read_text())
    frame,features,digest=load_data('multiscale_vit');assert digest==protocol['manifest_sha256']
    for value in features.values():value[2091:]=np.nan
    records=[]
    for arm in protocol['configuration_plan']:
        cfg=configuration(reference,arm);target=OUT/arm['name'];folds=[]
        for phase,fit,stop,outer,_ in phase_plans(frame,reference):
            evaluation=stop if phase=='validation' else outer
            for key in ['patient_group','split_group']:
                groups=[set(frame.iloc[ids][key]) for ids in [fit,stop,outer]]
                assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
            fit_mask=np.zeros(len(frame),dtype=bool);stop_mask=fit_mask.copy();fit_mask[fit]=True;stop_mask[stop]=True;ps=[]
            dump(OUT/'split_plans'/f'{phase}.json',{'fit_indices':fit.tolist(),'stop_indices':stop.tolist(),'outer_indices':outer.tolist()})
            for seed in range(3):
                dump(OUT/'state.json',{'status':'training','pid':os.getpid(),'arm':arm['name'],'phase':phase,'seed':seed,'external_test_evaluated':False})
                folder=target/phase/f'seed_{seed}'
                train_one(frame,features,digest,{**cfg,'seed':seed},folder,fit_mask,stop_mask,model_class=DistanceTaxonomyModel)
                c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False);ps.append(predict_cpu(c,features,evaluation))
            p=np.mean(ps,0);result={'phase':phase,'metrics':metrics(frame.iloc[evaluation].label,p)}
            np.savez_compressed(target/phase/'predictions.npz',indices=evaluation,p=p,pred=p.argmax(1));dump(target/phase/'metrics.json',result)
            if phase!='validation':
                np.savez_compressed(target/phase/'outer_predictions.npz',indices=evaluation,p=p,pred=p.argmax(1));folds.append(result)
        record=summarize_group(target/'validation',frame);record.update(name=arm['name'],candidate_id=arm['name'],distance_configuration=arm)
        record['cv']={'mean_macro_f1':float(np.mean([r['metrics']['macro_f1'] for r in folds])),
            'sd_macro_f1':float(np.std([r['metrics']['macro_f1'] for r in folds],ddof=1)),'folds':folds}
        record['paired_original_oof_vs_reference']=paired_oof(Path(reference['selected']['source_cv_summary']).parent,target,frame)
        record['development_gate_passed']=record['cv']['mean_macro_f1']>reference['cv']['mean_macro_f1'] and record['val']['macro_f1']>=reference['selected']['val']['macro_f1']
        records.append(record);dump(OUT/'leaderboard.json',records)
        print(json.dumps({'event':'distance_taxonomy_arm_complete','arm':arm['name'],'val_f1':record['val']['macro_f1'],
            'cv_f1':record['cv']['mean_macro_f1'],'development_gate_passed':record['development_gate_passed']}),flush=True)
    dump(DEST/'comparison.json',{'manifest_sha256':digest,'protocol':protocol,'candidates':records,'new_seed_fits':48,'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'cpu_auditing','pid':os.getpid(),'external_test_evaluated':False})
    audit(frame,features,digest,reference,records)
    eligible=[r for r in records if r['development_gate_passed']]
    if eligible:
        chosen=max(eligible,key=lambda r:r['cv']['mean_macro_f1'])
        dump(DEST/'selection_before_repeat.json',{'candidate':chosen,
            'frozen_validation_checkpoint_sha256':{str(seed):sha(path) for seed,path in enumerate(chosen['checkpoints'])},
            'requires_fresh_patient_confirmation':True,'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'new_seed_fits':48,
        'passing_development_gates':len(eligible),'external_test_evaluated':False})
    from report import report
    report()


def audit(frame,features,digest,reference,records):
    checked=[];diagnostics=[]
    for record in records:
        seen=[];scores=[]
        for phase,fit,stop,outer,_ in phase_plans(frame,reference):
            evaluation=stop if phase=='validation' else outer;tf=frame.iloc[fit];mapping=tf.groupby('disease').label.first();ps=[]
            for seed in range(3):
                folder=OUT/record['name']/phase/f'seed_{seed}';c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
                assert c['manifest_sha256']==digest and c['diseases']==sorted(tf.disease.unique())
                assert c['cfg']=={**configuration(reference,record['distance_configuration']),'seed':seed}
                assert c['dims']['disease_to_class']==[int(mapping[d]) for d in c['diseases']]
                for key,values in features.items():
                    z=values[fit];axes=tuple(range(z.ndim-1))
                    assert np.allclose(c['stats'][key]['mean'],z.mean(axes),atol=1e-6,rtol=1e-6)
                    assert np.allclose(c['stats'][key]['std'],np.maximum(z.std(axes),.05),atol=1e-6,rtol=1e-6)
                p=predict_cpu(c,features,stop);saved=np.load(folder/'val_predictions.npz')['p']
                assert np.allclose(p,saved,atol=5e-5,rtol=5e-5) and np.array_equal(p.argmax(1),saved.argmax(1))
                outer_p,detail=predict_cpu(c,features,evaluation,True);ps.append(outer_p)
                diagnostics.append({'arm':record['name'],'phase':phase,'seed':seed,**detail})
                checked.append({'arm':record['name'],'phase':phase,'seed':seed,'checkpoint_sha256':sha(folder/'model.pt')})
            p=np.mean(ps,0);saved=np.load(OUT/record['name']/phase/'predictions.npz')
            assert np.array_equal(saved['indices'],evaluation) and np.array_equal(saved['pred'],p.argmax(1))
            assert np.allclose(saved['p'],p,atol=5e-5,rtol=5e-5)
            actual=metrics(frame.iloc[evaluation].label,saved['p']);stored=json.loads((OUT/record['name']/phase/'metrics.json').read_text())['metrics']
            for key in ['accuracy','macro_f1','log_loss']:assert abs(actual[key]-stored[key])<1e-12
            if phase!='validation':seen.extend(outer.tolist());scores.append(actual['macro_f1'])
        assert sorted(seen)==np.flatnonzero(frame.split.eq('train')).tolist()
        assert abs(np.mean(scores)-record['cv']['mean_macro_f1'])<1e-12
        print(json.dumps({'event':'distance_taxonomy_cpu_audited','arm':record['name'],'heads':12}),flush=True)
    assert len(checked)==48
    dump(DEST/'distance_diagnostics.json',{'models':diagnostics})
    dump(DEST/'verification.json',{'passed':True,'manifest_sha256':digest,'actual_seed_heads_checked':48,'fits':checked,
        'checks':['All48fit-onlystatistics,diseasemaps,patientpartitions andstop/outerprobabilities restored onCPU',
            'All48actualLayerNorm,projection,boundedtangent,exponentialmap,prototypegeometry andlogits independentlyrebuilt withNumPy/arcoshformula',
            'Euclidean/Poincarecontrols at eachrank haveidenticalparameters andinitialclassifierweights; prototypepointshavevalidnorms',
            'Allensembles,classdecisions,foldmetrics andoriginaltrainingcoverage reconstructed; noexternalscoring']})


def main():
    torch.set_num_threads(4)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(OUT/'runner.pid').write_text(str(os.getpid()))
        try:run()
        except BaseException:
            dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc()});raise


if __name__=='__main__':main()
