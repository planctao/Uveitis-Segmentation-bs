"""Matched original-resolution and cached-upsampled fundus image inputs."""
from io import BytesIO
import hashlib
from PIL import Image,ImageOps
from torch.utils.data import Dataset
from torchvision import transforms as T


def crop_original(source):
    im=ImageOps.exif_transpose(source).convert('RGB');w,h=im.size
    return im.crop((int(.05*w),int(.05*h),int(.95*w),int(.95*h)))


def jpeg_roundtrip(image):
    buffer=BytesIO();image.save(buffer,format='JPEG',quality=95,subsampling=0);buffer.seek(0)
    with Image.open(buffer) as decoded:return decoded.convert('RGB')


class HighResolutionPairs(Dataset):
    def __init__(self,frame,size=768):
        self.frame=frame.reset_index(drop=True);self.size=size
        self.transform=T.Compose([T.Resize((size,size),antialias=True),T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])

    def __len__(self):return len(self.frame)

    def __getitem__(self,index):
        row=self.frame.iloc[index]
        with Image.open(row.path) as source:
            cropped=crop_original(source)
            reference=ImageOps.pad(cropped,(448,448),Image.Resampling.LANCZOS,color=(0,0,0))
            assert hashlib.sha256(reference.tobytes()).hexdigest()==row.pixel_sha256,'Original image no longer matches the fixed manifest pixels'
            high=ImageOps.pad(cropped,(self.size,self.size),Image.Resampling.LANCZOS,color=(0,0,0))
            high=jpeg_roundtrip(high)
        with Image.open(row.cache_path) as cached:control=cached.convert('RGB')
        return self.transform(high),self.transform(control),index
