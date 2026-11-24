"""Within-parent label smoothing from fit-only patient-balanced DINO disease centers."""
import numpy as np
import torch
import torch.nn.functional as F
from experiment import loss_fn


def build_transition(frame,global_features,diseases,mode,temperature=.2):
    assert mode in ['uniform','geometry'] and temperature>0
    # Every patient has equal influence within a disease, irrespective of image count.
    values=global_features.astype('float64');values/=np.maximum(np.linalg.norm(values,axis=1,keepdims=True),1e-12)
    centers=[];counts=[];parents=[]
    for name in diseases:
        selected=frame.disease.eq(name).to_numpy();patients=frame.loc[selected,'patient_group'].unique();patient_centers=[]
        for patient in patients:
            center=values[selected & frame.patient_group.eq(patient).to_numpy()].mean(0)
            patient_centers.append(center/max(np.linalg.norm(center),1e-12))
        center=np.mean(patient_centers,0);centers.append(center/max(np.linalg.norm(center),1e-12))
        counts.append(len(patients));labels=frame.loc[selected,'label'].unique();assert len(labels)==1;parents.append(int(labels[0]))
    centers=np.stack(centers);similarity=centers@centers.T;parents=np.array(parents);matrix=np.zeros_like(similarity)
    for i in range(len(diseases)):
        allowed=(parents==parents[i]) & (np.arange(len(diseases))!=i)
        if not allowed.any():matrix[i,i]=1;continue
        scores=similarity[i,allowed]/temperature if mode=='geometry' else np.zeros(int(allowed.sum()))
        weights=np.exp(scores-scores.max());matrix[i,allowed]=weights/weights.sum()
    return {'transition':matrix.astype('float32'),'centers':centers,'similarity':similarity,
        'patient_counts':np.array(counts),'parents':parents,'diseases':np.array(diseases),'temperature':np.array(temperature)}


def transport_penalty(logits,disease,weight,transition):
    logq=F.log_softmax(logits,dim=-1);valid=disease>=0
    target=transition[disease.clamp(min=0)]
    transported=-(target*logq).sum(-1)
    hard=F.cross_entropy(logits,disease,reduction='none',ignore_index=-1)
    return ((transported-hard)*weight*valid).mean()


class SmoothingObjective:
    def __init__(self,transition,witness_path=None):
        self.transition=torch.as_tensor(transition,dtype=torch.float32)
        self.witness_path=witness_path;self.witness_written=False

    def __call__(self,out,y,disease,weights,cfg):
        base=loss_fn(out,y,disease,weights,cfg);alpha=cfg['taxonomy_smoothing_alpha']
        transition=self.transition.to(device=out['subtype'].device,dtype=out['subtype'].dtype)
        penalty=transport_penalty(out['subtype'],disease,weights['leaf'],transition) if alpha else base.new_zeros(())
        total=base+cfg.get('aux',.6)*alpha*penalty
        if self.witness_path is not None and not self.witness_written:
            proof={key:value.detach().cpu().numpy() for key,value in {
                'logits':out['subtype'],'disease':disease,'y':y,'weight':weights['leaf'],
                'transition':transition,'patients':weights['patient_ids']}.items()}
            proof.update(base_loss=float(base.detach()),penalty=float(penalty.detach()),total_loss=float(total.detach()),
                alpha=alpha,aux=cfg.get('aux',.6))
            np.savez_compressed(self.witness_path,**proof);self.witness_written=True
        return total
