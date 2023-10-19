""" Dataset related classes """

import os

from PIL import Image
from torch.utils.data import Dataset


class Div2kDataset(Dataset):
    """Custom pytorch inherted dataset for loading images from source folder"""

    def __init__(self, data_dir="./DIV2K", transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.images = sorted(os.listdir(self.data_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(f"{self.data_dir}/{self.images[idx]}")

        if self.transform:
            return self.transform(img)

        return img
