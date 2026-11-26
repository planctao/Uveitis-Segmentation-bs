"""Locate failed bitwise saved-probability recovery across all48 frozen fits."""
import json,time
import numpy as np
import torch
from experiment import ROOT,dump
from kernel_residual_experiments import phase_plans,sha
from register_patch_query_data import load_all
from register_patch_query_experiments import restore,predict
from diagnose_patch_query_probability_replay import difference


def main():
    torch.set_num_threads(4);started=time.time();out=ROOT/'runs/register_patch_query'
    assert json.loads((out/'state.json').read_text())['status']=='failed'
    frozen=json.loads((out/'training_products_freeze_before_audit.json').read_text())['input_artifact_sha256']
    for path,expected in frozen.items():assert sha(path)==expected,path
    frame,features,bank,tail,digest=load_all()
    reference=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())
    plans={phase:(fit,stop,outer) for phase,fit,stop,outer,_ in phase_plans(frame,reference)}
    protocol=json.loads((out/'protocol.json').read_text());rows=[]
    for arm in protocol['configuration_plan']:
        for phase,(fit,stop,outer) in plans.items():
            for seed in range(3):
                folder=out/arm['name']/phase/f'seed_{seed}'
                c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
                saved=np.load(folder/'val_predictions.npz')['p']
                model,source=restore(c,tail);p=predict(model,source,features,bank,stop)
                row={'arm':arm['name'],'phase':phase,'seed':seed,'epoch':c['epoch'],'saved_vs_restored':difference(saved,p)}
                if not row['saved_vs_restored']['bitwise_equal']:
                    repeat=predict(model,source,features,bank,stop)
                    manual=predict(model,source,features,bank,stop,independent=True)
                    row['repeat_vs_first']=difference(p,repeat);row['saved_vs_independent']=difference(saved,manual)
                rows.append(row);print(json.dumps(row),flush=True)
                del model;torch.cuda.empty_cache()
    for path,expected in frozen.items():assert sha(path)==expected,path
    dump(ROOT/'results/register_patch_query_all_probability_replay_diagnosis.json',{'rows':rows,'fits_checked':len(rows),
        'bitwise_failures':sum(not row['saved_vs_restored']['bitwise_equal'] for row in rows),'all_original_products_unchanged':True,
        'new_classification_fits':0,'classification_scores_evaluated':0,'external_test_evaluated':False,'seconds':time.time()-started})


if __name__=='__main__':main()
