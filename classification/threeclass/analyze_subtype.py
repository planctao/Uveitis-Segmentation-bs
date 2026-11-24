"""Measure validation contribution of the subtype-to-class decision branch."""
import json
import numpy as np
import torch
from experiment import ROOT,Model,load_data,normalize,metrics,dump

@torch.inference_mode()
def main():
    torch.set_num_threads(4);frame,features,digest=load_data('vitb_320');mask=frame.split.eq('val').to_numpy();val=frame.loc[mask]
    selected=json.loads((ROOT/'runs/vitb_320/subtype_hierarchy/selection.json').read_text())['best']
    probabilities=[];removed=[];subtype=[];weights=[]
    for path in selected['checkpoints']:
        ckpt=torch.load(path,map_location='cpu',weights_only=False);assert ckpt['manifest_sha256']==digest
        model=Model(ckpt['cfg'],ckpt['dims'],len(ckpt['diseases'])).cuda();model.load_state_dict(ckpt['state_dict'],strict=True);model.eval()
        normalized,_=normalize({k:v[mask] for k,v in features.items()},None,ckpt['stats']);x={k:torch.from_numpy(v).cuda() for k,v in normalized.items()}
        pp=[];rr=[];ss=[];ww=[]
        for start in range(0,len(val),256):
            o=model({k:v[start:start+256] for k,v in x.items()})
            pp.append(o['p'].cpu().numpy());w=o['mixture_weights']
            if w.shape[1]==4:
                views=torch.stack([o['flat'].softmax(-1),o['pi'],o['pn']],1)
                no_fine=(views*w[:,:3,None]).sum(1)/w[:,:3].sum(1,keepdim=True).clamp(min=1e-7)
                rr.append(no_fine.cpu().numpy());ss.append(o['subtype_probability'].cpu().numpy());ww.append(w.cpu().numpy())
        probabilities.append(np.concatenate(pp))
        if rr:removed.append(np.concatenate(rr));subtype.append(np.concatenate(ss));weights.append(np.concatenate(ww))
        del model,x;torch.cuda.empty_cache()
    p=np.mean(probabilities,0);result={'full':metrics(val.label,p),'validation_only':True,'selection':selected['group']}
    if removed:
        result.update(with_subtype_branch_removed=metrics(val.label,np.mean(removed,0)),subtype_branch_only=metrics(val.label,np.mean(subtype,0)),mean_mixture_weights=np.mean(weights,axis=(0,1)).tolist(),note='Branch removal is a diagnostic of jointly trained weights; separately trained controls are in the ablation leaderboard.')
    dump(ROOT/'runs/vitb_320/subtype_hierarchy/contribution.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
