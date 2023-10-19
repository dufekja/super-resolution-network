import os

from PIL import Image
from torch.utils.data import Dataset

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
        
        return img