"""Coupled flip and mild photometric training views without cropping retinal content."""
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
from norm_adaptation_experiments import Pairs


class RetinalAugment:
    def __init__(self,mode='none'):
        assert mode in ['none','horizontal','horizontal_photometric']
        self.mode=mode
        self.jitter=T.ColorJitter(brightness=.1,contrast=.1,saturation=.05,hue=0)

    def __call__(self,image):
        if self.mode=='none':return image.copy()
        original=np.array(image.convert('RGB'),copy=True)
        if self.mode=='horizontal_photometric':
            pixels=np.array(self.jitter(Image.fromarray(original)),copy=True)
            background=original.max(-1)<=10
            pixels[background]=original[background]
        else:pixels=original.copy()
        if torch.rand(()).item()<.5:pixels=np.flip(pixels,axis=1).copy()
        return Image.fromarray(pixels)


class AugmentedPairs(Pairs):
    def __init__(self,frame,diseases,mode='none'):
        super().__init__(frame,diseases);self.augmentation=RetinalAugment(mode)

    def __getitem__(self,i):
        row=self.frame.iloc[i]
        with Image.open(row.cache_path) as source:
            transformed=self.augmentation(source.convert('RGB'))
            a,b=[t(transformed) for t in self.transforms]
        return a,b,int(row.label),int(self.disease[i]),float(self.weights[i]),int(self.patient[i]),i
