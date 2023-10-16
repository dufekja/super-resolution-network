import os
import random
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

class ImgTransform(object):
    """ Img transformer """
    def __init__(self, crop, scale, is_train=True, output='Tensor'):
        self.crop = crop
        self.scale = scale
        self.is_train = is_train
        self.output = output

    def __call__(self, img):

        if self.is_train:
            
            # crop HR image 
            left, top = random.randint(0, img.width - self.crop), random.randint(0, img.height - self.crop)
            right, bottom = left + self.crop, top + self.crop
    
            hr_img = img.crop((left, top, right, bottom))
        else:
            right, bottom = (img.width // self.scale) * self.scale, (img.height // self.scale) * self.scale
            hr_img = img.crop((0, 0, right, bottom))     
            
        # downscale hr image
        lr_img = hr_img.resize((hr_img.width // self.scale, hr_img.height // self.scale), Image.BICUBIC)
        
        assert lr_img.width * self.scale == hr_img.width
        assert lr_img.height * self.scale == hr_img.height
        assert (hr_img.width % self.scale, hr_img.height % self.scale) == (0, 0)
        
        # convert to tensor
        if self.output == 'Tensor':
            return pil_to_tensor(lr_img).type(torch.float), pil_to_tensor(hr_img).type(torch.float)
        
        return lr_img, hr_img

class Div2kDataset(Dataset):
    def __init__(self, dir='./data', transform=None):
        self.dir = dir
        self.transform = transform
        
        self.images = sorted([x for x in os.listdir(self.dir)])
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(f'{self.dir}/{self.images[idx]}')

        if self.transform:
            return self.transform(img)