"""Independent saved36fit output review; no raw-image or model replay."""
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch
from review_register_patch_query_selection_outputs import independent_metrics,verify_metric

ROOT=Path(__file__).resolve().parent
PROTOCOL=ROOT/'runs/large_lastblock_readout_coupling_saved_review/protocol.json'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def review(family='large_lastblock_readout_coupling'):
    assert family=='large_lastblock_readout_coupling'
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
    assert state['new_seed_fits']==comparison['new_seed_fits']==36
    assert verification['actual_seed_heads_checked']==verification['actual_training_witnesses_checked']==len(verification['fits'])==36
    assert verification['frozen_register_source_heads_checked']==len(verification['register_source_heads'])==12
    assert verification['reused_fusion_source_heads_checked']==len(verification['source_heads'])==24
    assert len(verification['fixed_fusion_phases_checked'])==12
    assert verification['only_adapter_and_linear_readout_trained']
    assert verification['adapter_parameters_per_joint_arm']==16384
    assert verification['original_ram_cache_and_frozen_source_verified']
    for section in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256','package_implementation_sha256','actual_development_image_sha256']:
        for name,expected in protocol[section].items():
            path=Path(name);check(path if path.is_absolute() else ROOT/path,expected)
    products=document(run/'training_products_freeze_before_audit.json')['input_artifact_sha256'];assert len(products)==180
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
    cache=document(run/'ram_activation_cache_before_any_fit.json')
    assert cache['mode']=='original' and cache['classification_fits_so_far']==0
    metadata=cache['metadata'];assert metadata['cache_storage_dtype']=='float32'
    assert metadata['indices']==np.flatnonzero(frame.split.isin(['train','val'])).tolist()
    assert metadata['all_numpy_cache_arrays_read_only']
    from register_multiquery_data import load_all
    loaded,raw_features,digest=load_all();assert digest==rule['manifest_sha256']
    checked=set()
    for row in verification['fits']:
        identity=(row['arm'],row['phase'],row['seed']);assert identity not in checked;checked.add(identity)
        folder=run/row['arm']/row['phase']/f"seed_{row['seed']}"
        check(folder/'model.pt',row['checkpoint_sha256']);check(folder/'training_input_witness.npz',row['training_witness_sha256'])
        checkpoint=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
        arm=next(a for a in protocol['configuration_plan'] if a['name']==row['arm'])
        config=arm['name'];fit=fitting[row['phase']]
        assert checkpoint['arm']==config and checkpoint['seed']==row['seed']
        assert np.array_equal(checkpoint['fit_indices'],fit)
        check(checkpoint['source_checkpoint'],checkpoint['source_checkpoint_sha256'])
        source=torch.load(checkpoint['source_checkpoint'],map_location='cpu',weights_only=False)
        leaves=len(source['diseases']);readout_parameters=257*leaves
        names={'head.flat.weight','head.flat.bias'}
        expected_names=['head.flat.weight','head.flat.bias']
        if config!='readout_only':
            names|={'adapter_a','adapter_b'}
            expected_names=['adapter_a','adapter_b']+expected_names
        expected_parameters=readout_parameters+(0 if config=='readout_only' else 16384)
        values=checkpoint['trainable_state_dict']
        assert set(values)==names and sum(v.numel() for v in values.values())==expected_parameters
        assert values['head.flat.weight'].shape==(leaves,256) and values['head.flat.bias'].shape==(leaves,)
        assert all(torch.isfinite(v).all() for v in values.values())
        if config!='readout_only':
            rank,output=(4,3072) if config=='attention_joint' else (8,1024)
            assert values['adapter_a'].shape==(rank,1024) and values['adapter_b'].shape==(output,rank)
        witness=row['witness'];assert set(witness['trainable_parameter_names'])==names
        assert witness['actual_images']==256 and witness['only_prescribed_adapter_and_readout_trainable']
        assert witness['source_zero_features_outputs_and_original_cuda_output_hashes_restored']
        with np.load(folder/'training_input_witness.npz') as first:
            ids=fit[np.random.default_rng(20294600+row['seed']).permutation(len(fit))[:256]]
            assert np.array_equal(first['indices'],ids) and np.array_equal(first['fit_indices'],fit)
            assert str(first['arm'])==config and int(first['seed'])==row['seed']
            for key,value in raw_features.items():
                digest=hashlib.sha256(np.ascontiguousarray(value[ids]).tobytes()).hexdigest()
                assert str(first['raw_sha256_'+key])==digest
            assert first['gradient'].shape==(expected_parameters,) and np.isfinite(first['gradient']).all()
            assert first['parameter_names'].tolist()==expected_names
            assert np.isfinite(first['loss']) and first['probability'].shape==(256,3)
            for size in [320,448]:
                assert str(first['rope_sha256_'+str(size)])==metadata['rotary_inputs'][str(size)]['sha256']
                assert len(str(first['cache_sha256_'+str(size)]))==64
            assert witness['actual_source_raw_and_ram_input_hashes_verified']
            assert witness['readout_gradients_finite_and_nonzero']
            assert witness['zero_adapter_a_and_nonzero_b_verified']==(config!='readout_only')
            for name in ['head.flat.weight','head.flat.bias']:
                value=source['state_dict'][name.removeprefix('head.')].detach().cpu().numpy()
                assert str(first['initial_sha256_'+name])==hashlib.sha256(value.tobytes()).hexdigest()
    assert len(checked)==36
    candidate_names=[a['name'] for a in protocol['configuration_plan']];assert len(candidate_names)==3
    assert len(comparison['candidates'])==6
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
        name=candidate['name'];mean,val=restored[name];arm=candidate['large_lastblock_readout_coupling_configuration']
        suffix='_m8replace' if name.endswith('_m8replace') else ''
        assert candidate['role']==arm['role']
        control_rows=[restored[n+suffix] for n in ['token_joint','readout_only']]
        threshold=max([protocol['minimum_cv_macro_f1']]+[r[0] for r in control_rows])
        validation_threshold=max([protocol['minimum_validation_macro_f1']]+[r[1]['macro_f1'] for r in control_rows])
        passed=(arm['role']=='sole_upgrade_candidate' and mean>threshold and val['macro_f1']>=validation_threshold)
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
        'classification_fits':36,'actual_task_witnesses':36,'training_products_rechecked':180,
        'saved_candidate_phases_metrics_recomputed':24,'two_matched_control_metric_phases_recomputed':16,'patient_and_duplicate_group_plans_checked':4,
        'fixed_fusion_phases_recomputed':12,'actual_first_raw_feature_witnesses_checked':36,
        'rope_hashes_per_training_witness_checked':2,'original_and_two_matched_control_gates_independently_checked':6,
        'gate_passers':len(eligible),'rows':rows,'input_sha256':inputs,'code_sha256':sha(__file__),
        'new_model_fits':0,'new_model_predictions':0,'new_candidate_scores':0,'external_test_evaluated':False,
        'scope_limit':'Independent saved24metricphases (16controlphases),12fusions,6M8-plus-two-control gates with controls never nominated,4patientplans,180products and36cached-raw-feature/taskwitnesses. Actual RAM input selection and task gradients rely on completed semantic audit; no raw-image/model replay or fresh-confirmation claim.'}


def main():
    target=ROOT/'results/research_large_lastblock_readout_coupling/independent_saved_output_review.json'
    assert not target.exists(),'Preserve prior completed reviews; version any recovery.'
    result=review()
    with target.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps({'review_passed':True,'gate_passers':result['gate_passers'],'inputs':len(result['input_sha256'])}),flush=True)


if __name__=='__main__':main()
