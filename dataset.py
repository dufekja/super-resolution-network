""" Dataset related classes """

import os

from PIL import Image
from torch.utils.data import Dataset


class Div2kDataset(Dataset):
    """ Custom pytorch dataset for loading images from source folder
    
    Args:
    data_dir -- directory containing images (default: ./DIV2K)
    transformer -- transformer class for image formatting before image return 
    """

    def __init__(self, data_dir="./DIV2K", transformer=None):
        self.data_dir = data_dir
        self.transformer = transformer

        self.images = sorted(os.listdir(self.data_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(f"{self.data_dir}/{self.images[idx]}")

        if self.transformer:
            return self.transformer(img)

        return img
