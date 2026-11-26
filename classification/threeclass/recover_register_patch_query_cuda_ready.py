"""Audit-only CUDA conversion recovery; all original48 fits and strict checks retained."""
import fcntl
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from experiment import ROOT,KEYS,normalize,loss_fn,metrics,dump,load_data
from kernel_residual_experiments import phase_plans,sha
from normal_large_residual_experiments import check_partitions
from register_patch_query_data import load_all
from register_patch_query import RegisterPatchQueryModel
from register_patch_query_witness import tensors_for,weights_for,write_witness,audit_witness
from audit_register_patch_query import independent_probability
from audit_large_token_evidence import audit_training_witness as audit_original_witness,independent_probability as source_independent
from finalize_large_token_evidence import restore_fusion_sources

OUT=ROOT/'runs/register_patch_query'
DEST=ROOT/'results/research_register_patch_query'
RECOVERY=OUT/'cuda_ready_audit_recovery'


def verify_frozen(protocol):
    assert json.loads((OUT/'protocol.json').read_text())==protocol
    for section in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
        for name,expected in protocol[section].items():
            path=Path(name);assert sha(path if path.is_absolute() else ROOT/path)==expected,name


def restore(c,tail):
    source=torch.load(c['source_checkpoint'],map_location='cpu',weights_only=False);assert sha(c['source_checkpoint'])==c['source_checkpoint_sha256']
    model=RegisterPatchQueryModel(source,tail,c['adapter_config'],20291700+c['seed'])
    expected={name for name,p in model.named_parameters() if p.requires_grad};assert set(c['query_state_dict'])==expected
    with torch.no_grad():
        for name,p in model.named_parameters():
            if p.requires_grad:p.copy_(c['query_state_dict'][name])
    return model.eval(),source


@torch.inference_mode()
def predict(model,source,features,bank,ids,device='cuda',independent=False):
    model=model.to(device).eval();result=[]
    for begin in range(0,len(ids),256):
        selected_ids=ids[begin:begin+256];base,selected,registers=tensors_for(features,bank,selected_ids,source,device)
        if independent:
            state={name:p.detach() for name,p in model.named_parameters() if p.requires_grad}
            raw={key:torch.from_numpy(features[key][selected_ids]).to(device) for key in features}
            tail_state={name:value for name,value in model.tail.state_dict().items()}
            out=independent_probability(source,tail_state,state,raw,selected,model.config)
        else:out=model(base,selected,registers)
        result.append(out['p'].cpu().numpy())
    return np.concatenate(result)




def run():
    recovery=json.loads((OUT/'audit_cuda_ready_protocol.json').read_text())
    for section in ['code_sha256','input_artifact_sha256']:
        for name,expected in recovery[section].items():assert sha(ROOT/name if section=='code_sha256' else name)==expected,name
    protocol=json.loads((OUT/'protocol.json').read_text());verify_frozen(protocol)
    frame,features,bank,tail,digest=load_all();assert digest==protocol['manifest_sha256']
    reference=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text());plans={};sources=[]
    for phase,fit,stop,outer,_ in phase_plans(frame,reference):
        check_partitions(frame,fit,stop,outer);plans[phase]=(fit,stop,outer)
        assert json.loads((OUT/'split_plans'/(phase+'.json')).read_text())=={'fit_indices':fit.tolist(),'stop_indices':stop.tolist(),'outer_indices':outer.tolist()}
        for seed in range(3):
            folder=ROOT/'runs/large_token_evidence/registers'/phase/f'seed_{seed}';source=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
            assert source['manifest_sha256']==digest and set(source['diseases'])==set(frame.iloc[fit].disease)
            for key,value in features.items():
                x=value[fit];axes=tuple(range(x.ndim-1));assert np.allclose(source['stats'][key]['mean'],x.mean(axes),atol=1e-6,rtol=1e-6)
                assert np.allclose(source['stats'][key]['std'],np.maximum(x.std(axes),.05),atol=1e-6,rtol=1e-6)
            witness=audit_original_witness(source,folder,features,frame,fit)
            sources.append({'phase':phase,'seed':seed,'checkpoint':str(folder/'model.pt'),'checkpoint_sha256':sha(folder/'model.pt'),'actual_original_fulltask_witness':witness})
    frozen={'new_seed_fits_so_far':0,'split_plan_sha256':{str(path):sha(path) for path in (OUT/'split_plans').glob('*.json')},
        'source_geometry_sha256':{str(ROOT/'runs/large_token_evidence/geometry'/phase/'geometry.npz'):sha(ROOT/'runs/large_token_evidence/geometry'/phase/'geometry.npz') for phase in plans}}
    assert json.loads((OUT/'geometry_freeze_before_all_fits.json').read_text())==frozen
    products=json.loads((OUT/'training_products_freeze_before_audit.json').read_text())['input_artifact_sha256']
    assert len(products)==240
    for path,expected in products.items():assert sha(path)==expected,path
    records=[];checked=[]
    for arm in protocol['configuration_plan']:
        config={key:arm[key] for key in ['kind','rank','alpha']};folds=[]
        for phase,(fit,stop,outer) in plans.items():
            dump(RECOVERY/'state.json',{'status':'independent_auditing','pid':os.getpid(),'arm':arm['name'],'phase':phase,'external_test_evaluated':False})
            ps=[];evaluation=stop if phase=='validation' else outer
            for seed in range(3):
                folder=OUT/arm['name']/phase/f'seed_{seed}';c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
                assert c['adapter_config']==config and c['seed']==seed
                history=json.loads((folder/'history.json').read_text());assert max(history,key=lambda r:(r['val']['macro_f1'],-r['val']['log_loss']))['epoch']==c['epoch']
                model,source=restore(c,tail);model=model.cuda();p=predict(model,source,features,bank,stop);saved=np.load(folder/'val_predictions.npz')['p']
                assert np.array_equal(p,saved)
                p=predict(model,source,features,bank,evaluation);manual=predict(model,source,features,bank,evaluation,independent=True)
                assert np.allclose(p,manual,atol=5e-5,rtol=5e-5) and np.array_equal(p.argmax(1),manual.argmax(1))
                witness=audit_witness(folder/'training_input_witness.npz',source,tail,features,bank,frame,fit,config,seed)
                checked.append({'arm':arm['name'],'phase':phase,'seed':seed,'checkpoint_sha256':sha(folder/'model.pt'),'training_witness_sha256':sha(folder/'training_input_witness.npz'),'witness':witness});ps.append(p)
                del model;torch.cuda.empty_cache()
            p=np.mean(ps,0);folder=OUT/arm['name']/phase;np.savez_compressed(folder/'predictions.npz',indices=evaluation,p=p,pred=p.argmax(1));score=metrics(frame.iloc[evaluation].label,p)
            if phase=='validation':val=score
            else:folds.append({'phase':phase,'metrics':score});np.savez_compressed(folder/'outer_predictions.npz',indices=evaluation,p=p,pred=p.argmax(1))
        row={'name':arm['name'],'candidate_id':arm['name'],'register_patch_query_configuration':arm,'val':val,'cv':{'folds':folds,'mean_macro_f1':float(np.mean([r['metrics']['macro_f1'] for r in folds]))}}
        row['development_gate_passed']=row['cv']['mean_macro_f1']>protocol['minimum_cv_macro_f1'] and val['macro_f1']>=protocol['minimum_validation_macro_f1'];records.append(row)
    _,base,base_digest=load_data('multiscale_vit');assert base_digest==digest
    for value in base.values():value[2091:]=np.nan
    b_sources=restore_fusion_sources(frame,base,digest,plans);assert len(b_sources)==24
    mixtures=[];phases=[]
    for source in records:
        folds=[];name=source['name']+'_m8replace'
        for phase,(fit,stop,outer) in plans.items():
            ids=stop if phase=='validation' else outer
            banks=[np.load(path/phase/'predictions.npz') for path in [ROOT/'runs/normal_subspace/zero_control',ROOT/'runs/normal_subspace/normal_rank32_full',OUT/source['name']]]
            assert all(np.array_equal(bank['indices'],ids) for bank in banks)
            a,b,t=[bank['p'] for bank in banks];p=(.1875*a+.5625*b+.25*t).astype('float32')
            manual=(.1875*a.astype('float64')+.5625*b.astype('float64')+.25*t.astype('float64')).astype('float32')
            assert np.allclose(p,manual,atol=2e-7,rtol=2e-7) and np.array_equal(p.argmax(1),manual.argmax(1))
            folder=OUT/name/phase;folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/'predictions.npz',indices=ids,p=p,pred=p.argmax(1));score=metrics(frame.iloc[ids].label,p)
            if phase=='validation':val=score
            else:folds.append({'phase':phase,'metrics':score});np.savez_compressed(folder/'outer_predictions.npz',indices=ids,p=p,pred=p.argmax(1))
            phases.append({'source':source['name'],'phase':phase,'images':len(ids)})
        row={**source,'name':name,'candidate_id':name,'source_register_patch_query_arm':source['name'],'m8_replacement_weight':.25,'val':val,'cv':{'folds':folds,'mean_macro_f1':float(np.mean([r['metrics']['macro_f1'] for r in folds]))}}
        row['development_gate_passed']=row['cv']['mean_macro_f1']>protocol['minimum_cv_macro_f1'] and val['macro_f1']>=protocol['minimum_validation_macro_f1'];mixtures.append(row)
    verify_frozen(protocol)
    for path,expected in products.items():assert sha(path)==expected
    for group in ['split_plan_sha256','source_geometry_sha256']:
        for path,expected in frozen[group].items():assert sha(path)==expected
    assert len(checked)==48 and len(phases)==16
    all_records=records+mixtures;eligible=[r for r in all_records if r['development_gate_passed']]
    dump(DEST/'comparison.json',{'manifest_sha256':digest,'protocol':protocol,'candidates':all_records,'new_seed_fits':48,'external_test_evaluated':False})
    dump(DEST/'verification.json',{'passed':True,'actual_seed_heads_checked':48,'actual_training_witnesses_checked':48,'fits':checked,'frozen_register_source_heads_checked':12,'register_source_heads':sources,
        'reused_fusion_source_heads_checked':24,'source_heads':b_sources,'fixed_fusion_phases_checked':phases,'only_query_adapters_trained':True,'audit_recovery_protocol_sha256':sha(OUT/'audit_cuda_ready_protocol.json'),'all48_saved_stop_probabilities_bitwise_identical':True,'external_test_evaluated':False})
    if eligible:
        winner=max(eligible,key=lambda r:r['cv']['mean_macro_f1']);name=winner.get('source_register_patch_query_arm',winner['name'])
        dump(DEST/'selection_before_repeat.json',{'candidate':winner,'requires_fresh_patient_confirmation':True,'checkpoint_sha256':{str(OUT/name/phase/f'seed_{seed}'/'model.pt'):sha(OUT/name/phase/f'seed_{seed}'/'model.pt') for phase in plans for seed in range(3)},'external_test_evaluated':False})
    dump(DEST/'freeze_integrity_after_training.json',{'passed':True,'all_original_sources_bank_tail_preprocessing_and_training_products_unchanged':True,'external_test_evaluated':False})
    dump(RECOVERY/'state.json',{'status':'completed','pid':os.getpid(),'actual_seed_heads_checked':48,'new_classification_fits_during_recovery':0,'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'new_seed_fits':48,'passing_development_gates':len(eligible),'external_test_evaluated':False})
    hypothesis=ROOT/'results/next_register_patch_query_hypothesis.json';draft=json.loads(hypothesis.read_text());dump(hypothesis,{**draft,'status':'completed_requires_confirmation' if eligible else 'completed_no_candidate_passed','classification_fits':48,'scores_evaluated':8,'passing_development_gates':len(eligible)})
    print(json.dumps({'event':'patch_query_complete','passing_development_gates':len(eligible)}),flush=True)


def main():
    torch.set_num_threads(4)
    with (RECOVERY/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except BaseException:
            dump(RECOVERY/'state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc()});raise


if __name__=='__main__':main()
