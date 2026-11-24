"""Check source immutability, one-view/two-resolutions coupling and patient metadata."""
import hashlib
import json
import numpy as np
from PIL import Image
import torch
from experiment import ROOT,load_data,dump
from norm_adaptation_experiments import Pairs
from retinal_training_augmentation import RetinalAugment,AugmentedPairs


def main():
    torch.set_num_threads(4)
    frame,_,digest=load_data('multiscale_vit');fit=frame[frame.split.eq('train')]
    ids=np.random.default_rng(20281213).choice(len(fit),8,replace=False);selected=fit.iloc[ids]
    diseases=sorted(fit.disease.unique());base=Pairs(selected,diseases);rows=[]
    for mode in ['none','horizontal','horizontal_photometric']:
        dataset=AugmentedPairs(selected,diseases,mode)
        for i in range(len(dataset)):
            path=selected.iloc[i].cache_path;before=hashlib.sha256(open(path,'rb').read()).hexdigest()
            with Image.open(path) as source:im=source.convert('RGB')
            untouched=np.array(im,copy=True)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(23+i);augmented=dataset.augmentation(im)
            assert np.array_equal(np.array(im),untouched) and augmented.size==im.size
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(23+i);actual=dataset[i]
            expected=[t(augmented) for t in dataset.transforms]
            assert torch.equal(actual[0],expected[0]) and torch.equal(actual[1],expected[1])
            assert actual[2:]==base[i][2:]
            assert actual[0].shape==(3,320,320) and actual[1].shape==(3,448,448)
            assert torch.isfinite(actual[0]).all() and torch.isfinite(actual[1]).all()
            if mode=='none':assert torch.equal(actual[0],base[i][0]) and torch.equal(actual[1],base[i][1])
            if mode=='horizontal':
                arr=np.array(augmented);assert np.array_equal(arr,untouched) or np.array_equal(arr,untouched[:,::-1])
            assert hashlib.sha256(open(path,'rb').read()).hexdigest()==before
        black=Image.fromarray(np.zeros((24,24,3),dtype='uint8'))
        for seed in range(4):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed);assert not np.array(RetinalAugment(mode)(black)).any()
        rows.append({'mode':mode,'actual_train_images_checked':len(dataset)})
    # Photometric changes restore exactly the original dark background before optional flip.
    pixels=np.zeros((32,32,3),dtype='uint8');pixels[8:24,8:24]=[160,70,30]
    for seed in range(5):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);image=np.array(RetinalAugment('horizontal_photometric')(Image.fromarray(pixels)))
        assert not image[:8].any() and not image[24:].any()
    dump(ROOT/'verification_retinal_training_augmentation.json',{'passed':True,'manifest_sha256':digest,'configurations':rows,
        'checks':['Trainingonlyimage sources unchangedbyte-for-byte andinputPILobjectnotmutated',
            'OnestochasticPILviewfeedsboth320/448resolutions;seededreplayexact',
            'NoaugmentationexactoriginalPairs;horizontalmodeexactidentityorflipwithoutcropping',
            'Labels,diseases,patientweights,patientIDs andlocalindicesunchanged',
            'Photometricbackgroundrestorationandall-blackinvariance; finitecorrectshapes'],
        'status':'Input pipeline verified; no augmentation model fit claimed.'})
    print('Retinal training views: source integrity, metadata, background and coupled resolutions verified.',flush=True)


if __name__=='__main__':main()
