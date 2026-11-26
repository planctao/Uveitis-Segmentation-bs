"""Compare shared-query execution contexts without fitting or scoring."""
import json
import numpy as np
import torch
from experiment import ROOT,dump
from kernel_residual_experiments import phase_plans
from register_patch_query_data import load_all
from register_patch_query_experiments import restore,predict
from register_patch_query_witness import tensors_for
from diagnose_patch_query_probability_replay import difference


def main():
    torch.set_num_threads(4);frame,features,bank,tail,digest=load_all()
    reference=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())
    _,fit,stop,outer,_=next(iter(phase_plans(frame,reference)))
    folder=ROOT/'runs/register_patch_query/shared_rank4/validation/seed_1'
    c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False);saved=np.load(folder/'val_predictions.npz')['p'];rows=[]
    for context in ['restored','train_before_predict','gradient_warmup','state_reload','broadcast_materialized']:
        model,source=restore(c,tail);model=model.cuda()
        if context=='train_before_predict':model.train()
        if context=='gradient_warmup':
            ids=fit[:256];base,selected,registers=tensors_for(features,bank,ids,source,'cuda')
            result=model.train()(base,selected,registers);result['subtype'].square().mean().backward()
        if context=='state_reload':
            state={name:value.detach().cpu().clone() for name,value in model.state_dict().items()};model.load_state_dict(state)
        if context=='broadcast_materialized':
            original=model.down.forward
            model.down.forward=lambda value: original(value.contiguous())
        p=predict(model,source,features,bank,stop)
        rows.append({'context':context,'saved_vs_prediction':difference(saved,p)})
        print(json.dumps(rows[-1]),flush=True)
        del model;torch.cuda.empty_cache()
    dump(ROOT/'results/register_patch_query_shared_execution_diagnosis.json',{'rows':rows,'classification_fits':0,'classification_scores':0,'external_test_evaluated':False})


if __name__=='__main__':main()
