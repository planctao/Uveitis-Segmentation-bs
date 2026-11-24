"""Spatial neighbor contrast at3x3/7x7, each matched to its plain grid control."""
import fcntl
import json
import os
from pathlib import Path
import traceback
import numpy as np
import torch
from experiment import ROOT, Model, load_data, train_one, normalize, predict, summarize_group, metrics, dump
from foreground_data import load_foreground
from kernel_residual_experiments import phase_plans, sha
from milestone2 import paired_oof

OUT = ROOT / 'runs/spatial_graph'


def load_arm(arm):
    return load_data('multiscale_vit') if arm['grid'] == 3 else load_foreground('fine7_unmasked')


def arm_config(reference, arm):
    cfg = {**reference['selected']['config'], 'region_graph_steps': arm['steps']}
    if arm['grid'] == 7:cfg['use_region_mask'] = False
    return cfg


@torch.inference_mode()
def predict_cpu(c, features, indices, diagnostics=False):
    model = Model(c['cfg'], c['dims'], len(c['diseases'])).eval()
    model.load_state_dict(c['state_dict'], strict=True)
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())
    probabilities, removed = [], []
    magnitudes = []
    def measure(module, inputs, output):
        delta = (output - inputs[0]).square().mean((1,2)).sqrt()
        scale = inputs[0].square().mean((1,2)).sqrt().clamp(min=1e-8)
        magnitudes.extend((delta / scale).tolist())
    handle = model.region_graph.register_forward_hook(measure) if diagnostics else None
    for begin in range(0,len(indices),128):
        selected = indices[begin:begin+128]
        x,_ = normalize({key:value[selected] for key,value in features.items()},None,c['stats'])
        inputs = {key:torch.from_numpy(value) for key,value in x.items()}
        out = model(inputs);probabilities.append(out['p'].numpy())
        if diagnostics:
            original_cfg=model.cfg;model.cfg={**model.cfg,'region_graph_steps':0}
            removed.append(model(inputs)['p'].numpy());model.cfg=original_cfg
    if handle:handle.remove()
    return np.concatenate(probabilities), ({'branch_off_p':np.concatenate(removed),
        'mean_relative_token_change':float(np.mean(magnitudes)),
        'relative_token_change_quantiles':np.quantile(magnitudes,[0,.1,.5,.9,1]).tolist(),
        'outgoing_weight_norm':float(model.region_graph.outgoing.weight.norm()),
        'incoming_weight_norm':float(model.region_graph.incoming.weight.norm())} if diagnostics else None)


def run():
    protocol=json.loads((OUT/'protocol.json').read_text())
    for path in ['verification_spatial_graph.json','verification_foreground_features.json','results/research_foreground_multiscale/verification.json']:
        assert json.loads((ROOT/path).read_text())['passed']
    reference=json.loads((ROOT/protocol['reference']).read_text())
    fine_control=next(row for row in json.loads((ROOT/'results/research_foreground_multiscale/comparison.json').read_text())['candidates'] if row['name']=='fine7_unmasked')
    records=[]
    for arm in protocol['configuration_plan']:
        frame,features,digest=load_arm(arm);config=arm_config(reference,arm)
        group=OUT/arm['name']/'validation'
        for seed in range(3):
            dump(OUT/'state.json',{'status':'formal_training','pid':os.getpid(),'arm':arm['name'],'seed':seed,'external_test_evaluated':False})
            train_one(frame,features,digest,{**config,'seed':seed},group/f'seed_{seed}')
        record=summarize_group(group,frame);record.update(name=arm['name'],candidate_id='spatial_'+arm['name'],graph_configuration=arm)
        folds=[]
        for phase,fit,stop,outer,_ in phase_plans(frame,reference)[1:]:
            fit_mask=np.zeros(len(frame),dtype=bool);stop_mask=fit_mask.copy();fit_mask[fit]=True;stop_mask[stop]=True
            target=OUT/arm['name']/phase;ps=[]
            for seed in range(3):
                dump(OUT/'state.json',{'status':'patient_cv_training','pid':os.getpid(),'arm':arm['name'],'phase':phase,'seed':seed,'external_test_evaluated':False})
                train_one(frame,features,digest,{**config,'seed':seed},target/f'seed_{seed}',fit_mask,stop_mask)
                c=torch.load(target/f'seed_{seed}/model.pt',map_location='cpu',weights_only=False)
                model=Model(c['cfg'],c['dims'],len(c['diseases'])).cuda().eval();model.load_state_dict(c['state_dict'],strict=True)
                values,_=normalize({key:value[outer] for key,value in features.items()},None,c['stats'])
                x={key:torch.from_numpy(value).cuda() for key,value in values.items()}
                p,_=predict(model,x);ps.append(p)
                del model,x;torch.cuda.empty_cache()
            p=np.mean(ps,0);np.savez_compressed(target/'outer_predictions.npz',indices=outer,p=p,pred=p.argmax(1))
            row={'phase':phase,'metrics':metrics(frame.iloc[outer].label,p)};dump(target/'metrics.json',row);folds.append(row)
        record['cv']={'mean_macro_f1':float(np.mean([row['metrics']['macro_f1'] for row in folds])),
            'sd_macro_f1':float(np.std([row['metrics']['macro_f1'] for row in folds],ddof=1)),'folds':folds}
        control=fine_control if arm['grid']==7 else {'val':reference['selected']['val'],'cv':reference['cv']}
        record['matching_plain_grid_control']={'name':'fine7_unmasked' if arm['grid']==7 else 'milestone5','val':control['val'],'cv':control['cv']}
        record['development_gate_passed']=(record['cv']['mean_macro_f1']>reference['cv']['mean_macro_f1'] and record['val']['macro_f1']>=reference['selected']['val']['macro_f1'] and record['cv']['mean_macro_f1']>control['cv']['mean_macro_f1'] and record['val']['macro_f1']>=control['val']['macro_f1'])
        source_cv=Path(reference['selected']['source_cv_summary']).parent
        record['paired_original_oof_vs_reference']=paired_oof(source_cv,OUT/arm['name'],frame)
        if arm['grid']==7:record['paired_original_oof_vs_matching_grid']=paired_oof(ROOT/'runs/foreground_multiscale/fine7_unmasked',OUT/arm['name'],frame)
        records.append(record);dump(OUT/'leaderboard.json',records)
        print(json.dumps({'event':'spatial_graph_arm_complete','arm':arm['name'],'val_f1':record['val']['macro_f1'],'cv_f1':record['cv']['mean_macro_f1']}),flush=True)
        del features
    destination=ROOT/'results/research_spatial_graph'
    dump(destination/'comparison.json',{'manifest_sha256':digest,'protocol':protocol,'candidates':records,'new_seed_fits':48,'external_test_evaluated':False})
    audit_results(reference,protocol,records)
    eligible=[row for row in records if row['development_gate_passed']]
    if eligible:dump(destination/'selection_before_repeat.json',{'candidate':max(eligible,key=lambda row:row['cv']['mean_macro_f1']),'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'new_seed_fits':48,'passing_development_gates':len(eligible),'external_test_evaluated':False})
    from report import report
    report()


def audit_results(reference,protocol,records):
    checked=[];diagnostics=[]
    for arm,record in zip(protocol['configuration_plan'],records):
        frame,features,digest=load_arm(arm);seen=[];scores=[]
        for phase,fit,stop,outer,_ in phase_plans(frame,reference):
            evaluation=stop if phase=='validation' else outer
            for key in ['patient_group','split_group']:
                groups=[set(frame.iloc[ids][key]) for ids in [fit,stop,outer]]
                assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
            tf=frame.iloc[fit];ps=[];removed=[]
            for seed in range(3):
                folder=OUT/arm['name']/phase/f'seed_{seed}'
                c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
                assert c['cfg']=={**arm_config(reference,arm),'seed':seed}
                assert c['manifest_sha256']==digest and c['diseases']==sorted(tf.disease.unique())
                mapping=tf.groupby('disease').label.first()
                assert c['dims']['disease_to_class']==[int(mapping[name]) for name in c['diseases']]
                for key,value in features.items():
                    z=value[fit];axes=tuple(range(z.ndim-1))
                    assert np.allclose(c['stats'][key]['mean'],z.mean(axes),atol=1e-6,rtol=1e-6)
                    assert np.allclose(c['stats'][key]['std'],np.maximum(z.std(axes),.05),atol=1e-6,rtol=1e-6)
                stop_p,_=predict_cpu(c,features,stop)
                saved=np.load(folder/'val_predictions.npz')['p']
                assert np.allclose(stop_p,saved,atol=5e-5,rtol=5e-5) and np.array_equal(stop_p.argmax(1),saved.argmax(1))
                p,detail=predict_cpu(c,features,evaluation,diagnostics=True)
                ps.append(p);removed.append(detail.pop('branch_off_p'))
                diagnostics.append({'arm':arm['name'],'phase':phase,'seed':seed,**detail})
                checked.append({'arm':arm['name'],'phase':phase,'seed':seed})
            p=np.mean(ps,0)
            actual=metrics(frame.iloc[evaluation].label,p)
            if phase=='validation':assert abs(actual['macro_f1']-record['val']['macro_f1'])<1e-12
            else:
                saved=np.load(OUT/arm['name']/phase/'outer_predictions.npz')
                assert np.array_equal(saved['indices'],outer) and np.allclose(saved['p'],p,atol=5e-5,rtol=5e-5)
                assert np.array_equal(saved['pred'],p.argmax(1))
                actual=metrics(frame.iloc[evaluation].label,saved['p'])
                stored=json.loads((OUT/arm['name']/phase/'metrics.json').read_text())['metrics']
                for key in ['accuracy','macro_f1','log_loss']:assert abs(actual[key]-stored[key])<1e-12
                seen.extend(outer.tolist());scores.append(actual['macro_f1'])
            dump(OUT/arm['name']/phase/'branch_diagnostic.json',{'branch_on':actual,
                'branch_off':metrics(frame.iloc[evaluation].label,np.mean(removed,0)),
                'limitation':'Disabling a jointly trained branch does not replace the independently trained plain grid control.'})
        assert sorted(seen)==np.flatnonzero(frame.split.eq('train')).tolist()
        assert abs(record['cv']['mean_macro_f1']-np.mean(scores))<1e-12
        print(json.dumps({'event':'spatial_graph_arm_cpu_audited','arm':arm['name'],'actual_heads':12}),flush=True)
    assert len(checked)==48
    destination=ROOT/'results/research_spatial_graph'
    dump(destination/'branch_diagnostics.json',{'models':diagnostics})
    dump(destination/'verification.json',{'passed':True,'manifest_sha256':digest,'actual_seed_heads_checked':48,'fits':checked,
        'checks':['Every fixed graph/grid combination completed all12actual seed heads',
            'All fitting/stopping/outer patient and duplicate groups disjoint; feature statistics and disease maps independently rebuilt',
            'Every actual stopping and outer probability independently restored onCPU',
            'All class decisions, ensemble metrics and original outer image coverage reconstructed',
            'Actual message parameter norms, token changes and jointlytrainedbranch-off diagnostics recorded',
            'Independent plain grid control comparison retained separately from branch-off diagnostics']})


def main():
    torch.set_num_threads(4)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(OUT/'runner.pid').write_text(str(os.getpid()))
        dump(OUT/'implementation.json',{'code_sha256':{name:sha(ROOT/name) for name in ['spatial_graph_experiments.py','spatial_graph.py','experiment.py']}})
        try:run()
        except BaseException:
            dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc()});raise


if __name__=='__main__':main()
