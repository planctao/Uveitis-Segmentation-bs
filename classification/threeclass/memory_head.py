"""Class-balanced patient memory with leave-one-patient-out training queries."""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class PatientMemory(nn.Module):
    def __init__(self,dimension,count,cfg):
        super().__init__();self.cfg=cfg
        if count<3:raise ValueError('Patient memory needs a prototype for every class')
        self.register_buffer('bank',torch.zeros(count,dimension))
        self.register_buffer('labels',torch.zeros(count,dtype=torch.long))
        self.register_buffer('patient_ids',torch.zeros(count,dtype=torch.long))
        self.register_buffer('global_mean',torch.zeros(dimension))
        self.register_buffer('global_std',torch.ones(dimension))
        self.log_scale=nn.Parameter(torch.tensor(math.log(cfg.get('memory_scale',10.))))

    @torch.no_grad()
    def initialize(self,standardized_global,labels,patient_ids,mean,std):
        z=torch.as_tensor(standardized_global,dtype=torch.float32)
        mean=torch.as_tensor(mean,dtype=torch.float32);std=torch.as_tensor(std,dtype=torch.float32)
        q=F.normalize(z*std+mean,dim=-1)
        labels=np.asarray(labels);patient_ids=np.asarray(patient_ids)
        groups=sorted(set(zip(patient_ids.tolist(),labels.tolist())))
        if len(groups)!=len(self.bank):raise ValueError('Memory bank allocation does not match training patients')
        vectors=[];info=[]
        for index,(pid,label) in enumerate(groups):
            mask=(patient_ids==pid)&(labels==label)
            vectors.append(F.normalize(q[torch.from_numpy(mask)].mean(0),dim=0))
            info.append({'prototype_index':index,'patient_token':int(pid),'label':int(label),'images':int(mask.sum())})
        if set(labels.tolist())!={0,1,2}:raise ValueError('All three classes must be present in the training memory')
        self.bank.copy_(torch.stack(vectors).to(self.bank.device))
        self.labels.copy_(torch.tensor([g[1] for g in groups],device=self.bank.device))
        self.patient_ids.copy_(torch.tensor([g[0] for g in groups],device=self.bank.device))
        self.global_mean.copy_(mean.to(self.bank.device));self.global_std.copy_(std.to(self.bank.device))
        return info

    def forward(self,standardized_global,query_patient_ids=None):
        q=F.normalize(standardized_global*self.global_std+self.global_mean,dim=-1)
        scale=self.log_scale.clamp(0,math.log(50.)).exp()
        similarities=q@self.bank.T*scale
        allowed=torch.ones_like(similarities,dtype=torch.bool)
        if query_patient_ids is not None:
            allowed &= query_patient_ids[:,None]!=self.patient_ids[None,:]
        class_scores=[]
        k=min(self.cfg.get('memory_topk',5),len(self.bank))
        for label in range(3):
            valid=allowed & (self.labels[None,:]==label)
            values=similarities.masked_fill(~valid,-1e4).topk(k,dim=1).values
            retained=values>-5000;count=retained.sum(1)
            score=torch.logsumexp(values.masked_fill(~retained,-1e4),dim=1)-count.clamp(min=1).float().log()
            class_scores.append(torch.where(count>0,score,torch.full_like(score,-1e4)))
        scores=torch.stack(class_scores,dim=1)
        scores=torch.where(allowed.any(1,keepdim=True),scores,torch.zeros_like(scores))
        return scores.softmax(-1),scores
