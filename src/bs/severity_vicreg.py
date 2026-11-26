"""Author-formula VICReg loss and an FA encoder adaptation."""
import torch
from torch import nn
import torch.nn.functional as F

from bs.severity_swav_images import dinov3_convnext_tiny


def vicreg_loss(x,y):
    if x.ndim!=2 or x.shape!=y.shape or x.shape[0]<2:
        raise ValueError('VICReg needs matching [batch>=2, feature] matrices')
    # Autocast may supply float16 embeddings; statistics need float32 accuracy.
    if x.dtype in [torch.float16,torch.bfloat16]:x=x.float()
    if y.dtype in [torch.float16,torch.bfloat16]:y=y.float()
    invariance=F.mse_loss(x,y)
    x=x-x.mean(0);y=y-y.mean(0)
    std_x=torch.sqrt(x.var(0,unbiased=True)+1e-4)
    std_y=torch.sqrt(y.var(0,unbiased=True)+1e-4)
    variance=(F.relu(1-std_x).mean()+F.relu(1-std_y).mean())/2
    cov_x=x.T@x/(len(x)-1);cov_y=y.T@y/(len(y)-1)
    offdiag=~torch.eye(x.shape[1],dtype=torch.bool,device=x.device)
    covariance=(cov_x[offdiag].square().sum()+cov_y[offdiag].square().sum())/x.shape[1]
    loss=25*invariance+25*variance+covariance
    return loss,dict(invariance=invariance,variance=variance,covariance=covariance,
        mean_std=(std_x.mean()+std_y.mean())/2)


class ImageVICReg(nn.Module):
    def __init__(self,weights):
        super().__init__()
        self.encoder=dinov3_convnext_tiny(pretrained=False)
        self.encoder.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
        self.expander=nn.Sequential(nn.Linear(768,512),nn.BatchNorm1d(512),nn.ReLU(),
            nn.Linear(512,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Linear(512,128,bias=False))

    def forward(self,images):
        features=self.encoder.forward_features(images)['x_norm_clstoken']
        return features,self.expander(features)
