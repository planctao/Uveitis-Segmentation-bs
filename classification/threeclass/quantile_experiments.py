"""Predeclared distributional regional quantile study and complete CPU reconstruction."""
import fcntl
import json
import os
from pathlib import Path
import traceback
import numpy as np
import torch
from experiment import ROOT, load_data, normalize, train_one, summarize_group, metrics, dump
from kernel_residual_experiments import phase_plans, sha
from milestone2 import paired_oof
from quantile_summary import QuantileModel
from verify_quantile_summary import independent_quantile

OUT=ROOT/'runs/quantile_summary'
DEST=ROOT/'results/research_quantile_summary'


def configuration(reference,arm):
    return {**reference['selected']['config'],'quantile_rank':arm['rank'],'quantile_kind':arm['kind']}


@torch.inference_mode()
def predict_cpu(c,features,indices,diagnostics=False):
    assert len(indices) and np.max(indices)<2091
    model=QuantileModel(c['cfg'],c['dims'],len(c['diseases'])).eval();model.load_state_dict(c['state_dict'],strict=True)
    ps=[];off=[];relative=[];errors=[];spread=[]
    for start in range(0,len(indices),256):
        ids=indices[start:start+256]
        x,_=normalize({key:value[ids] for key,value in features.items()},None,c['stats'])
        assert all(np.isfinite(v).all() for v in x.values())
        tx={key:torch.from_numpy(value) for key,value in x.items()};out=model(tx);ps.append(out['p'].numpy())
        if diagnostics:
            expected=independent_quantile(model.quantile_summary,out['quantile_input'].numpy(),x.get('region_mask') if c['cfg'].get('use_region_mask',True) else None)
            for key,value in zip(['quantile_delta','quantile_summary','quantile_values'],expected):
                errors.append(float(np.max(np.abs(out[key].numpy()-value))))
                assert np.allclose(out[key].numpy(),value,atol=3e-5,rtol=3e-5)
            spread.append((out['quantile_summary'][:,2]-out['quantile_summary'][:,0]).abs().mean(1).numpy())
            relative.append((out['quantile_delta'].norm(dim=-1)/out['quantile_original'].norm(dim=-1).clamp(min=1e-7)).numpy())
            model.cfg['quantile_disabled']=True;off.append(model(tx)['p'].numpy());model.cfg.pop('quantile_disabled')
    if not diagnostics:return np.concatenate(ps)
    return np.concatenate(ps),{'mean_relative_hidden_delta':float(np.concatenate(relative).mean()),
        'max_relative_hidden_delta':float(np.concatenate(relative).max()),'max_independent_branch_error':max(errors),
        'readout_weight_norm':float(model.quantile_summary.readout.weight.norm()),
        'projection_weight_norm':float(model.quantile_summary.down.weight.norm()),
        'mean_summary_interquartile_spread':float(np.concatenate(spread).mean()),'branch_off_p':np.concatenate(off)}


def run():
    protocol=json.loads((OUT/'protocol.json').read_text());assert json.loads((ROOT/'verification_quantile_summary.json').read_text())['passed']
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
                train_one(frame,features,digest,{**cfg,'seed':seed},folder,fit_mask,stop_mask,model_class=QuantileModel)
                c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False);ps.append(predict_cpu(c,features,evaluation))
            p=np.mean(ps,0);result={'phase':phase,'metrics':metrics(frame.iloc[evaluation].label,p)}
            np.savez_compressed(target/phase/'predictions.npz',indices=evaluation,p=p,pred=p.argmax(1));dump(target/phase/'metrics.json',result)
            if phase!='validation':
                np.savez_compressed(target/phase/'outer_predictions.npz',indices=evaluation,p=p,pred=p.argmax(1));folds.append(result)
        record=summarize_group(target/'validation',frame);record.update(name=arm['name'],candidate_id=arm['name'],quantile_configuration=arm)
        record['cv']={'mean_macro_f1':float(np.mean([r['metrics']['macro_f1'] for r in folds])),
            'sd_macro_f1':float(np.std([r['metrics']['macro_f1'] for r in folds],ddof=1)),'folds':folds}
        record['paired_original_oof_vs_reference']=paired_oof(Path(reference['selected']['source_cv_summary']).parent,target,frame)
        record['development_gate_passed']=record['cv']['mean_macro_f1']>reference['cv']['mean_macro_f1'] and record['val']['macro_f1']>=reference['selected']['val']['macro_f1']
        records.append(record);dump(OUT/'leaderboard.json',records)
        print(json.dumps({'event':'quantile_arm_complete','arm':arm['name'],'val_f1':record['val']['macro_f1'],
            'cv_f1':record['cv']['mean_macro_f1'],'development_gate_passed':record['development_gate_passed']}),flush=True)
    dump(DEST/'comparison.json',{'manifest_sha256':digest,'protocol':protocol,'candidates':records,'new_seed_fits':48,'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'cpu_auditing','pid':os.getpid(),'external_test_evaluated':False})
    audit(frame,features,digest,reference,records)
    eligible=[r for r in records if r['development_gate_passed']]
    if eligible:dump(DEST/'selection_before_repeat.json',{'candidate':max(eligible,key=lambda r:r['cv']['mean_macro_f1']),
        'requires_fresh_patient_confirmation':True,'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'new_seed_fits':48,'passing_development_gates':len(eligible),'external_test_evaluated':False})
    from report import report
    report()


def audit(frame,features,digest,reference,records):
    checked=[];diagnostics=[]
    for record in records:
        seen=[];scores=[]
        for phase,fit,stop,outer,_ in phase_plans(frame,reference):
            evaluation=stop if phase=='validation' else outer;tf=frame.iloc[fit];mapping=tf.groupby('disease').label.first();ps=[];off=[]
            for seed in range(3):
                folder=OUT/record['name']/phase/f'seed_{seed}';c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
                assert c['manifest_sha256']==digest and c['diseases']==sorted(tf.disease.unique())
                assert c['cfg']=={**configuration(reference,record['quantile_configuration']),'seed':seed}
                assert c['dims']['disease_to_class']==[int(mapping[d]) for d in c['diseases']]
                for key,values in features.items():
                    assert np.isnan(values[2091:]).all();z=values[fit];axes=tuple(range(z.ndim-1))
                    assert np.allclose(c['stats'][key]['mean'],z.mean(axes),atol=1e-6,rtol=1e-6)
                    assert np.allclose(c['stats'][key]['std'],np.maximum(z.std(axes),.05),atol=1e-6,rtol=1e-6)
                p=predict_cpu(c,features,stop);saved=np.load(folder/'val_predictions.npz')['p']
                assert np.allclose(p,saved,atol=5e-5,rtol=5e-5) and np.array_equal(p.argmax(1),saved.argmax(1))
                p,detail=predict_cpu(c,features,evaluation,True);ps.append(p);off.append(detail.pop('branch_off_p'))
                assert detail['mean_relative_hidden_delta']>0 and detail['readout_weight_norm']>0
                diagnostics.append({'arm':record['name'],'phase':phase,'seed':seed,**detail})
                checked.append({'arm':record['name'],'phase':phase,'seed':seed,'checkpoint_sha256':sha(folder/'model.pt')})
            p=np.mean(ps,0);saved=np.load(OUT/record['name']/phase/'predictions.npz')
            assert np.array_equal(saved['indices'],evaluation) and np.array_equal(saved['pred'],p.argmax(1)) and np.allclose(saved['p'],p,atol=5e-5,rtol=5e-5)
            actual=metrics(frame.iloc[evaluation].label,saved['p']);stored=json.loads((OUT/record['name']/phase/'metrics.json').read_text())['metrics']
            for key in ['accuracy','macro_f1','log_loss']:assert abs(actual[key]-stored[key])<1e-12
            dump(OUT/record['name']/phase/'branch_diagnostic.json',{'branch_on':actual,'branch_off':metrics(frame.iloc[evaluation].label,np.mean(off,0)),
                'limitation':'Disabling jointly trained quantile summaries does not replace independently trained M5 control.'})
            if phase!='validation':seen.extend(outer.tolist());scores.append(actual['macro_f1'])
        assert sorted(seen)==np.flatnonzero(frame.split.eq('train')).tolist() and abs(np.mean(scores)-record['cv']['mean_macro_f1'])<1e-12
        print(json.dumps({'event':'quantile_cpu_audited','arm':record['name'],'heads':12}),flush=True)
    assert len(checked)==48
    dump(DEST/'branch_diagnostics.json',{'models':diagnostics})
    dump(DEST/'verification.json',{'passed':True,'manifest_sha256':digest,'actual_seed_heads_checked':48,'fits':checked,'checks':[
        'Four predeclared rank16/32 mean/quantile regional summaries completed twelve actual heads each',
        'Fit/stop/outer patient and duplicate groups checked; original fit-only normalization and disease maps reconstructed',
        'Every stopping and outer probability, ensemble decision and metric restored on CPU with exact original outer coverage',
        'Every actual masked regional distributional summary and residual independently reconstructed with NumPy/SciPy',
        'Trained nonzero readouts, projection norms, summary interquartile spread, relative residuals and branch-off diagnostics retained; no external scoring']})


def main():
    torch.set_num_threads(4)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(OUT/'runner.pid').write_text(str(os.getpid()))
        dump(OUT/'implementation.json',{'code_sha256':{name:sha(ROOT/name) for name in
            ['quantile_summary.py','verify_quantile_summary.py','quantile_experiments.py','experiment.py']}})
        try:run()
        except BaseException:
            dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc()});raise


if __name__=='__main__':main()
