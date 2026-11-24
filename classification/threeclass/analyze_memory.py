"""Validation-only diagnostics for whether the learned gate uses patient memory."""
import argparse
import json
import numpy as np
import torch
from experiment import ROOT,Model,load_data,normalize,metrics,dump

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',default='memory');a=parser.parse_args();torch.set_num_threads(4)
    frame,features,digest=load_data('vitb_320');mask=frame.split.eq('val').to_numpy();val=frame.loc[mask]
    directory=ROOT/'runs/vitb_320'/a.phase
    records=json.loads((directory/'leaderboard.json').read_text());reports=[]
    for record in records:
        if not record['config']['arch'].startswith('memory_'):continue
        outputs={'p':[],'parametric_probability':[],'memory_probability':[],'memory_gate':[]}
        for path in record['checkpoints']:
            ckpt=torch.load(path,map_location='cpu',weights_only=False);assert ckpt['manifest_sha256']==digest
            model=Model(ckpt['cfg'],ckpt['dims'],len(ckpt['diseases'])).cuda();model.load_state_dict(ckpt['state_dict'],strict=True);model.eval()
            normalized,_=normalize({k:v[mask] for k,v in features.items()},None,ckpt['stats'])
            data={k:torch.from_numpy(v).cuda() for k,v in normalized.items()};pieces={k:[] for k in outputs}
            for start in range(0,len(val),256):
                out=model({k:v[start:start+256] for k,v in data.items()})
                for key in pieces:pieces[key].append(out[key].cpu().numpy())
            for key in outputs:outputs[key].append(np.concatenate(pieces[key]))
            del model,data;torch.cuda.empty_cache()
        arrays={k:np.mean(v,axis=0) for k,v in outputs.items()};gate=arrays['memory_gate'][:,0]
        summary={'group':record['group'],'full':metrics(val.label,arrays['p']),'parametric_branch':metrics(val.label,arrays['parametric_probability']),'memory_branch':metrics(val.label,arrays['memory_probability']),'mean_gate':float(gate.mean()),'gate_quantiles':np.quantile(gate,[0,.25,.5,.75,1]).tolist(),'validation_only':True,'interpretation':'The parametric and memory branches were jointly trained; this is a contribution diagnostic, not an independently tuned baseline.'}
        out=__import__('pathlib').Path(record['checkpoints'][0]).parent.parent
        dump(out/'val_memory_diagnostics.json',summary);np.savez_compressed(out/'val_memory_diagnostics.npz',**arrays)
        reports.append(summary)
    dump(directory/'diagnostics.json',reports)
    for report in reports:print(json.dumps({'group':report['group'],'full_f1':report['full']['macro_f1'],'parametric_f1':report['parametric_branch']['macro_f1'],'memory_f1':report['memory_branch']['macro_f1'],'mean_gate':report['mean_gate']}),flush=True)

if __name__=='__main__':main()
