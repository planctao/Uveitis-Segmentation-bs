"""Training-only description of spatial evidence in the retained normal branch."""
import numpy as np
import torch
from experiment import ROOT,load_data,normalize,dump
from normal_subspace import NormalSuppressedModel
from kernel_residual_experiments import sha


def main():
    torch.set_num_threads(4)
    frame,features,digest=load_data('multiscale_vit');ids=np.flatnonzero(frame.split.eq('train'))
    c=torch.load(ROOT/'runs/normal_subspace/normal_rank32_full/validation/seed_0/model.pt',map_location='cpu',weights_only=False)
    model=NormalSuppressedModel(c['cfg'],c['dims'],len(c['diseases'])).eval();model.load_state_dict(c['state_dict'],strict=True)
    captured={}
    model.region_gate.register_forward_hook(lambda m,a,o:captured.update(scores=o))
    model.encoder[0].register_forward_pre_hook(lambda m,a:captured.update(summary=a[0]))
    rows=[]
    with torch.inference_mode():
        for begin in range(0,len(ids),128):
            batch=ids[begin:begin+128];data,_=normalize({k:v[batch] for k,v in features.items()},None,c['stats']);x={k:torch.from_numpy(v) for k,v in data.items()}
            p=model(x)['p'];attention=captured['scores'].softmax(1);projected=model.transform(x)['regions']
            centered=projected-projected.mean(1,keepdim=True)
            entropy=-(attention*attention.log()).sum(1).mean(1)/np.log(9)
            uniform={**x,'regions':x['regions'].mean(1,keepdim=True).expand(-1,9,-1)}
            q=model(uniform)['p'];js=.5*((p*(p.clamp(min=1e-7).log()-((p+q)/2).clamp(min=1e-7).log())).sum(1)+(q*(q.clamp(min=1e-7).log()-((p+q)/2).clamp(min=1e-7).log())).sum(1))
            for j,index in enumerate(batch):rows.append({'patient':frame.iloc[index].patient_group,'label':int(frame.iloc[index].label),'attention_entropy':float(entropy[j]),'centered_spatial_energy':float(centered[j].square().mean()),'all_region_energy':float(projected[j].square().mean()),'uniform_region_js':float(js[j]),'uniform_region_changes_prediction':bool(p[j].argmax()!=q[j].argmax())})
    import pandas as pd
    table=pd.DataFrame(rows);groups=table.groupby(['patient','label']).mean(numeric_only=True).reset_index();summary={}
    for label in range(3):summary[str(label)]={k:float(groups.loc[groups.label.eq(label),k].mean()) for k in ['attention_entropy','centered_spatial_energy','all_region_energy','uniform_region_js','uniform_region_changes_prediction']}
    dump(ROOT/'results/normal_branch_diagnostics.json',{'manifest_sha256':digest,'scope':'Train images only, retained trained seed0 normal branch, equal patient-label means. In-sample descriptive diagnostic, no classification-generalization or lesion-localization claim. Uniform-region replacement is an input intervention, not a physiological causal experiment.','summary':summary,'patient_label_groups':len(groups),'images':len(ids),'code_sha256':sha(ROOT/'normal_branch_diagnostics.py'),'external_test_evaluated':False})
    print(summary,flush=True)


if __name__=='__main__':main()
