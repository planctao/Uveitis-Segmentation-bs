"""Independent completed-output review for the preregistered direct patch-evidence study."""
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch
from review_register_patch_query_selection_outputs import independent_metrics,verify_metric

ROOT=Path(__file__).resolve().parent
PROTOCOL=ROOT/'runs/class_patch_log_moment_saved_review/protocol.json'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def review(family='class_patch_log_moment'):
    assert family=='class_patch_log_moment'
    run=ROOT/'runs'/family;result=ROOT/'results'/('research_'+family)
    assert (run/'state.json').exists(),'Only completed studies can be reviewed; study has not started.'
    state=json.loads((run/'state.json').read_text())
    assert state['status']=='completed','Never read active checkpoints.'
    rule=json.loads(PROTOCOL.read_text());assert sha(__file__)==rule['code_sha256']
    inputs={str(PROTOCOL):sha(PROTOCOL),str(Path(__file__).resolve()):sha(__file__)}
    def check(path,expected=None):
        path=Path(path);actual=sha(path)
        if expected is not None:assert actual==expected,str(path)
        if str(path) in inputs:assert inputs[str(path)]==actual
        inputs[str(path)]=actual
    def document(path):
        check(path);return json.loads(Path(path).read_text())
    for name,expected in rule['frozen_artifacts'].items():check(name,expected)
    check(ROOT/'metadata/manifest.csv',rule['manifest_sha256']);frame=pd.read_csv(ROOT/'metadata/manifest.csv')
    protocol=document(run/'protocol.json');comparison=document(result/'comparison.json')
    verification=document(result/'verification.json');integrity=document(result/'freeze_integrity_after_training.json')
    assert comparison['protocol']==protocol and verification['passed'] and integrity['passed']
    assert state['new_seed_fits']==comparison['new_seed_fits']==24
    assert verification['actual_seed_heads_checked']==verification['actual_training_witnesses_checked']==len(verification['fits'])==24
    assert verification['frozen_register_source_heads_checked']==len(verification['register_source_heads'])==12
    assert verification['reused_fusion_source_heads_checked']==len(verification['source_heads'])==24
    assert len(verification['fixed_fusion_phases_checked'])==8
    assert verification['only_patch_log_moment_readouts_and_bias_trained']
    assert verification['original_shared64_patch_bank_and_frozen_source_verified']
    for section in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
        for name,expected in protocol[section].items():
            path=Path(name);check(path if path.is_absolute() else ROOT/path,expected)
    products=document(run/'training_products_freeze_before_audit.json')['input_artifact_sha256'];assert len(products)==120
    for name,expected in products.items():check(name,expected)
    phases={};fitting={};base={};outer=[]
    for phase in ['validation','fold_0','fold_1','fold_2']:
        plan=document(run/'split_plans'/(phase+'.json'))
        source_plan=document(ROOT/'runs/register_patch_query/split_plans'/(phase+'.json'))
        arrays={key:np.asarray(plan[key],dtype=int) for key in ['fit_indices','stop_indices','outer_indices']}
        assert all(np.array_equal(value,source_plan[key]) for key,value in arrays.items())
        for key in ['patient_group','split_group']:
            groups=[set(frame.iloc[value][key]) for value in arrays.values()]
            assert all(not groups[i].intersection(groups[j]) for i in range(3) for j in range(i+1,3))
        assert frame.iloc[arrays['fit_indices']].split.eq('train').all()
        if phase=='validation':
            assert np.array_equal(arrays['fit_indices'],np.flatnonzero(frame.split.eq('train')))
            assert np.array_equal(arrays['stop_indices'],np.flatnonzero(frame.split.eq('val')))
        else:
            assert all(frame.iloc[value].split.eq('train').all() for value in arrays.values());outer.extend(arrays['outer_indices'].tolist())
        fitting[phase]=arrays['fit_indices'];phases[phase]=arrays['stop_indices'] if phase=='validation' else arrays['outer_indices']
        base[phase]={}
        for name,folder in [('rawB','normal_subspace/zero_control'),('normalB','normal_subspace/normal_rank32_full')]:
            path=ROOT/'runs'/folder/phase/'predictions.npz';check(path)
            with np.load(path) as saved:
                assert np.array_equal(saved['indices'],phases[phase]);base[phase][name]=saved['p'].copy()
    assert sorted(outer)==np.flatnonzero(frame.split.eq('train')).tolist()
    banks={name:np.load(ROOT/'runs/register_patch_query_bank/arrays'/(name+'.npy'),mmap_mode='r')
        for name in ['query_input','query_raw','patch_keys','patch_values','patch_indices','postattention']}
    checked=set()
    for row in verification['fits']:
        identity=(row['arm'],row['phase'],row['seed']);assert identity not in checked;checked.add(identity)
        folder=run/row['arm']/row['phase']/f"seed_{row['seed']}"
        check(folder/'model.pt',row['checkpoint_sha256']);check(folder/'training_input_witness.npz',row['training_witness_sha256'])
        checkpoint=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
        arm=next(a for a in protocol['configuration_plan'] if a['name']==row['arm'])
        config={key:arm[key] for key in ['pooling','statistic']};fit=fitting[row['phase']]
        assert checkpoint['adapter_config']==config and checkpoint['seed']==row['seed']
        assert np.array_equal(checkpoint['fit_indices'],fit)
        check(checkpoint['source_checkpoint'],checkpoint['source_checkpoint_sha256'])
        names={'patch_readout','patch_bias'}
        assert set(checkpoint['trainable_state_dict'])==names
        witness=row['witness'];assert set(witness['trainable_parameter_names'])==names
        assert witness['actual_images']==256 and witness['original_cuda_outputs_sha256_restored'] and witness['frozen_head_tail_verified']
        with np.load(folder/'training_input_witness.npz') as first:
            ids=fit[np.random.default_rng(20293500+row['seed']).permutation(len(fit))[:256]]
            assert np.array_equal(first['indices'],ids) and np.array_equal(first['fit_indices'],fit)
            assert json.loads(str(first['config']))==config and int(first['seed'])==row['seed']
            for key,value in banks.items():
                digest=hashlib.sha256(np.ascontiguousarray(value[ids]).tobytes()).hexdigest()
                assert str(first['bank_sha256_'+key])==digest
    assert len(checked)==24
    candidate_names=[a['name'] for a in protocol['configuration_plan']];assert len(candidate_names)==2
    assert len(comparison['candidates'])==4
    assert {r['name'] for r in comparison['candidates']}==set(candidate_names+[n+'_m8replace' for n in candidate_names])
    restored={};rows=[];probabilities={}
    for candidate in comparison['candidates']:
        name=candidate['name'];folds=[]
        for phase,ids in phases.items():
            path=run/name/phase/'predictions.npz';check(path)
            with np.load(path) as saved:
                assert np.array_equal(saved['indices'],ids);p=saved['p'].copy();assert np.array_equal(saved['pred'],p.argmax(1))
            actual=independent_metrics(frame.iloc[ids].label,p);probabilities[(name,phase)]=p
            if phase=='validation':verify_metric(candidate['val'],actual);val=actual
            else:
                verify_metric(next(r['metrics'] for r in candidate['cv']['folds'] if r['phase']==phase),actual)
                folds.append(actual['macro_f1'])
        mean=float(np.mean(folds));assert abs(mean-candidate['cv']['mean_macro_f1'])<1e-12
        restored[name]=(mean,val)
    for candidate in comparison['candidates']:
        name=candidate['name'];mean,val=restored[name];arm=candidate['class_patch_log_moment_configuration']
        cv_limits=[protocol['minimum_cv_macro_f1']];val_limits=[protocol['minimum_validation_macro_f1']]
        suffix='_m8replace' if name.endswith('_m8replace') else ''
        for control_key,control_section,control_family in [('matched_mean','frozen_mean_controls','class_query_patch_evidence'),('matched_variance','frozen_variance_controls','class_patch_dispersion')]:
            control_name=arm[control_key]+suffix
            control=next(r for r in protocol[control_section] if r['name']==control_name)
            control_folds=[]
            for phase,ids in phases.items():
                path=ROOT/'runs'/control_family/control_name/phase/'predictions.npz';check(path)
                with np.load(path) as saved:
                    assert np.array_equal(saved['indices'],ids)
                    actual=independent_metrics(frame.iloc[ids].label,saved['p'])
                if phase=='validation':verify_metric(control['val'],actual);control_val=actual
                else:
                    verify_metric(next(r['metrics'] for r in control['cv']['folds'] if r['phase']==phase),actual)
                    control_folds.append(actual['macro_f1'])
            control_cv=float(np.mean(control_folds));assert abs(control_cv-control['cv']['mean_macro_f1'])<1e-12
            cv_limits.append(control_cv);val_limits.append(control_val['macro_f1'])
        passed=mean>max(cv_limits) and val['macro_f1']>=max(val_limits)
        assert passed==candidate['development_gate_passed']
        rows.append({'candidate':name,'cv_macro_f1':mean,'validation_macro_f1':val['macro_f1'],
            'validation_accuracy':val['accuracy'],'development_gate_passed':passed})
    for name in candidate_names:
        for phase in phases:
            a,b=base[phase]['rawB'],base[phase]['normalB'];adapted=probabilities[(name,phase)];p=probabilities[(name+'_m8replace',phase)]
            assert np.array_equal(p,(.1875*a+.5625*b+.25*adapted).astype('float32'))
            independent=(.1875*a.astype('float64')+.5625*b.astype('float64')+.25*adapted.astype('float64')).astype('float32')
            assert np.allclose(p,independent,atol=2e-7,rtol=2e-7) and np.array_equal(p.argmax(1),independent.argmax(1))
    eligible=[r for r in comparison['candidates'] if r['development_gate_passed']]
    assert len(eligible)==state['passing_development_gates']
    selection=result/'selection_before_repeat.json'
    if eligible:
        chosen=document(selection);assert chosen['candidate']==max(eligible,key=lambda r:r['cv']['mean_macro_f1'])
        for name,expected in chosen['checkpoint_sha256'].items():check(name,expected)
    else:assert not selection.exists()
    for name,expected in inputs.items():assert sha(name)==expected,name
    return {'review_passed':True,'created_at_utc':datetime.now(timezone.utc).isoformat(),'family':family,
        'classification_fits':24,'actual_task_witnesses':24,'training_products_rechecked':120,
        'saved_candidate_phases_metrics_recomputed':16,'matched_mean_control_saved_phases_recomputed':16,'matched_variance_control_saved_phases_recomputed':16,'patient_and_duplicate_group_plans_checked':4,
        'fixed_fusion_phases_recomputed':8,'actual_first_input_patch_arrays_checked':24,
        'cache_arrays_per_training_witness_checked':6,'original_and_both_matched_control_gates_independently_checked':4,
        'gate_passers':len(eligible),'rows':rows,'input_sha256':inputs,'code_sha256':sha(__file__),
        'new_model_fits':0,'new_model_predictions':0,'new_candidate_scores':0,'external_test_evaluated':False,
        'scope_limit':'Independent saved16candidate+16matchedmean+16matchedvariancemetricphases/8fusions/4M8-and-both-matched-control-gates,4patientplans,120products and24sixarraytaskinputwitnesses. Existing semantic model/task audit checked, not rerun. Selecteddevelopment only; no fresh/causal/novelty/budget-matching claim or external scoring.'}


def main():
    target=ROOT/'results/research_class_patch_log_moment/independent_saved_output_review.json'
    assert not target.exists(),'Preserve prior completed reviews; version any recovery.'
    result=review()
    with target.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps({'review_passed':True,'gate_passers':result['gate_passers'],'inputs':len(result['input_sha256'])}),flush=True)


if __name__=='__main__':main()
