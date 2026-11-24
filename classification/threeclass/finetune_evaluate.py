"""Evaluate a validation-selected adapted-backbone candidate after selection."""
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from finetune import Fine,Images,predict
from experiment import ROOT,metrics,dump
from evaluate import aggregate,cluster_accuracy_ci

def main():
    torch.set_num_threads(4)
    selection=json.loads((ROOT/'runs/vitb_320/finetune/selection.json').read_text())
    out=ROOT/'results/vitb_320/finetune';out.mkdir(parents=True,exist_ok=True)
    dump(out/'selection_before_test.json',selection)
    if not selection['improved_over_frozen']:
        dump(out/'decision.json',{'decision':'Keep frozen candidate; validation fine-tuning score did not improve. No additional external evaluation.'});return
    group=selection['best'];frame=pd.read_csv(ROOT/'metadata/manifest.csv')
    digest=__import__('hashlib').sha256((ROOT/'metadata/manifest.csv').read_bytes()).hexdigest()
    assert selection['manifest_sha256']==digest
    all_prob={s:[] for s in ['val','test_independent','test_ningbo']}
    for path in group['checkpoints']:
        ckpt=torch.load(path,map_location='cpu',weights_only=False)
        assert ckpt['manifest_sha256']==digest
        model=Fine(ckpt,ckpt['adapt_blocks']).cuda()
        expected={k for k,p in model.backbone.named_parameters() if p.requires_grad}
        assert set(ckpt['backbone_adapted'])==expected,'All and only adapted parameters must be restored'
        restored=model.backbone.load_state_dict(ckpt['backbone_adapted'],strict=False)
        assert not restored.unexpected_keys and not expected & set(restored.missing_keys)
        for split in all_prob:
            df=frame.loc[frame.split.eq(split)].reset_index(drop=True)
            loader=DataLoader(Images(df,ckpt['diseases']),batch_size=32,num_workers=4,pin_memory=True)
            all_prob[split].append(predict(model,loader))
        del model,loader;torch.cuda.empty_cache()
    results={}
    for split,probs in all_prob.items():
        p=np.mean(probs,0);g=np.zeros((len(p),0),dtype='float32');pred=p.argmax(1)
        df=frame.loc[frame.split.eq(split)].reset_index(drop=True)
        result={'image':metrics(df.label,p,pred),'image_accuracy_patient_cluster_ci95':cluster_accuracy_ci(df,pred)}
        image=df[['path','patient_group','eye_group','disease','label']].copy()
        for i,name in enumerate(['infectious','noninfectious','normal']):image['p_'+name]=p[:,i]
        image['prediction']=pred;image.to_csv(out/f'{split}_images.csv',index=False)
        for key in ['patient_group','eye_group']:
            af,ap,_,excluded=aggregate(df,p,g,key);apr=ap.argmax(1);name='patient' if key=='patient_group' else 'eye'
            result[name]=metrics(af.label,ap,apr);result[name]['excluded_mixed_label_groups']=excluded
            af['prediction']=apr
            for i,cls in enumerate(['infectious','noninfectious','normal']):af['p_'+cls]=ap[:,i]
            af.to_csv(out/f'{split}_{name}.csv',index=False)
        results[split]=result
    assert abs(results['val']['image']['macro_f1']-group['val']['macro_f1'])<.01,'Reconstructed adapted predictions must reproduce validation selection'
    dump(out/'metrics.json',{'selection':group,'results':results,'status':'Exploratory adaptation candidate; first milestone CV-selected primary remains separately preserved.'})
    print(json.dumps({k:v['image'] for k,v in results.items()}),flush=True)

if __name__=='__main__':main()
