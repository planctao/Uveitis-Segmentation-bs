"""Replayable per-image, per-epoch retinal training views, independent of workers."""
import multiprocessing
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF
from retinal_training_augmentation import AugmentedPairs


class EpochAugmentedPairs(AugmentedPairs):
    def __init__(self,frame,diseases,seed,mode='horizontal_photometric'):
        super().__init__(frame,diseases,mode)
        self.manifest_indices=np.asarray(frame.index,dtype='int64')
        self.seed=int(seed)
        self.epoch=multiprocessing.Value('q',0)

    def set_epoch(self,epoch):
        with self.epoch.get_lock():self.epoch.value=int(epoch)

    def image_seed(self,index):
        return 20461124+self.seed*1000003+int(self.epoch.value)*10007+int(self.manifest_indices[index])

    def __getitem__(self,index):
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(self.image_seed(index))
            return super().__getitem__(index)


def independently_rebuild_view(image,seed):
    """Replay ColorJitter's documented random order with independent PIL operations."""
    original=np.array(image.convert('RGB'),copy=True)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(int(seed))
        order=torch.randperm(4).tolist()
        brightness=float(torch.empty(1).uniform_(.9,1.1))
        contrast=float(torch.empty(1).uniform_(.9,1.1))
        saturation=float(torch.empty(1).uniform_(.95,1.05))
        result=Image.fromarray(original)
        for operation in order:
            if operation==0:result=TF.adjust_brightness(result,brightness)
            elif operation==1:result=TF.adjust_contrast(result,contrast)
            elif operation==2:result=TF.adjust_saturation(result,saturation)
        pixels=np.array(result,copy=True)
        background=original.max(-1)<=10
        pixels[background]=original[background]
        flipped=bool(torch.rand(()).item()<.5)
        if flipped:pixels=pixels[:,::-1].copy()
    return Image.fromarray(pixels),dict(brightness=brightness,contrast=contrast,saturation=saturation,flipped=flipped,order=order)
