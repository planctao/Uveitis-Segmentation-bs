"""Patient-isolated image or centroid contrastive objectives, without new parameters."""
import torch
import torch.nn.functional as F


def patient_contrastive_loss(embedding,target,patient,weight,mode,temperature=.15):
    if mode not in ['exclude_patient','patient_centroid'] or temperature<=0:
        raise ValueError('Unknown patient contrastive mode or nonpositive temperature')
    embedding=F.normalize(embedding,dim=-1)
    if mode=='patient_centroid':
        keys,inverse=torch.unique(torch.stack([patient,target],dim=1),dim=0,return_inverse=True)
        centers=torch.zeros(len(keys),embedding.shape[1],dtype=embedding.dtype,device=embedding.device)
        centers=centers.index_add(0,inverse,embedding)
        # Normalizing the sum equals normalizing the mean and gives each patient-target one candidate.
        candidates=F.normalize(centers,dim=-1)
        source_patient,source_target=keys[:,0],keys[:,1]
    else:
        candidates=embedding;source_patient,source_target=patient,target
        inverse=torch.arange(len(target),device=target.device)
    allowed=(source_patient[:,None]!=source_patient[None,:]) & (source_target[:,None]>=0) & (source_target[None,:]>=0)
    positives=allowed & (source_target[:,None]==source_target[None,:])
    count=positives.sum(1);valid=count>0
    if not valid.any():return embedding.sum()*0
    scores=candidates@candidates.T/temperature
    # Invalid rows have no positive anchors; the finite floor also keeps their unused softmax defined.
    scores=scores.masked_fill(~allowed,-1e4)
    per_anchor=-(F.log_softmax(scores,dim=1)*positives).sum(1)/count.clamp(min=1)
    image_valid=valid[inverse]
    return (per_anchor[inverse][image_valid]*weight[image_valid]).mean()
