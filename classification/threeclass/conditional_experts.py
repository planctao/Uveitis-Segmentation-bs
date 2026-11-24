"""Independent conditional binary MLPs on a frozen DINO-taxonomy representation."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score,log_loss
from experiment import Head,sample_weights


def tasks(labels,order):
    y=np.asarray(labels)
    if order=='cascade_i':return [(np.ones(len(y),dtype=bool),(y==0).astype(int)),(y!=0,(y==1).astype(int))]
    if order=='cascade_n':return [(np.ones(len(y),dtype=bool),(y!=2).astype(int)),(y!=2,(y==0).astype(int))]
    raise ValueError('Unknown binary order')


def gates_from_probability(p,order):
    p=np.asarray(p)
    if order=='cascade_i':return np.stack([p[:,0],p[:,1]/np.maximum(p[:,1]+p[:,2],1e-7)],axis=1)
    if order=='cascade_n':return np.stack([p[:,0]+p[:,1],p[:,0]/np.maximum(p[:,0]+p[:,1],1e-7)],axis=1)
    raise ValueError('Unknown binary order')


def standardize_embedding(fit,others,stats=None):
    if stats is None:stats={'mean':fit.mean(0).astype('float32'),'std':np.maximum(fit.std(0),.05).astype('float32')}
    return [np.clip((x-stats['mean'])/stats['std'],-10,10).astype('float32') for x in [fit,*others]],stats


@torch.inference_mode()
def predict_expert(checkpoint,embedding):
    cfg=checkpoint['cfg'];head=Head(embedding.shape[1],cfg['hidden'],2,cfg['dropout']).eval()
    head.load_state_dict(checkpoint['state_dict'],strict=True)
    return head(torch.from_numpy(embedding)).softmax(-1)[:,1].numpy()


def fit_expert(fit_frame,stop_frame,fit_embedding,stop_embedding,order,stage,cfg,seed):
    torch.manual_seed(20261124+seed);np.random.seed(20261124+seed)
    fitting,target=tasks(fit_frame.label,order)[stage];stopping,stop_target=tasks(stop_frame.label,order)[stage]
    assert fitting.any() and stopping.any() and set(target[fitting])=={0,1} and set(stop_target[stopping])=={0,1}
    x=torch.from_numpy(fit_embedding[fitting]).cuda();y=torch.from_numpy(target[fitting]).long().cuda()
    vx=torch.from_numpy(stop_embedding[stopping]).cuda()
    weights=sample_weights(fit_frame,target,fitting)[fitting];weight=torch.from_numpy(weights).cuda()
    head=Head(x.shape[1],cfg['hidden'],2,cfg['dropout']).cuda();optimizer=torch.optim.AdamW(head.parameters(),lr=cfg['lr'],weight_decay=cfg['wd'])
    history=[];best=None;wait=0
    for epoch in range(cfg['epochs']):
        head.train();permutation=torch.randperm(len(x),device='cuda');total=0.
        for start in range(0,len(x),256):
            ix=permutation[start:start+256];optimizer.zero_grad(set_to_none=True)
            loss=(F.cross_entropy(head(x[ix]),y[ix],reduction='none')*weight[ix]).mean()
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5);optimizer.step();total+=float(loss.detach())*len(ix)
        with torch.inference_mode():
            head.eval();p=head(vx).softmax(-1).cpu().numpy();prediction=p.argmax(1)
        f1=float(f1_score(stop_target[stopping],prediction,average='macro',labels=[0,1],zero_division=0))
        scoring_probability=p.astype('float64');scoring_probability/=scoring_probability.sum(1,keepdims=True)
        cross_entropy=float(log_loss(stop_target[stopping],scoring_probability,labels=[0,1]))
        history.append({'epoch':epoch+1,'train_loss':total/len(x),'stop_macro_f1':f1,'stop_log_loss':cross_entropy})
        key=(f1,-cross_entropy)
        if best is None or key>best:
            best=key;wait=0
            checkpoint={'cfg':cfg,'order':order,'stage':stage,'seed':seed,'epoch':epoch+1,
                'state_dict':{key:value.detach().cpu().clone() for key,value in head.state_dict().items()},
                'stopping':{'macro_f1':f1,'log_loss':cross_entropy},'fit_images':int(fitting.sum()),'stop_images':int(stopping.sum()),
                'fitting_relative_indices':np.flatnonzero(fitting).tolist(),'stopping_relative_indices':np.flatnonzero(stopping).tolist(),
                'patient_class_weights':weights}
        else:wait+=1
        if wait>=cfg['patience']:break
    del head,x,y,vx,weight;torch.cuda.empty_cache()
    return checkpoint,history
