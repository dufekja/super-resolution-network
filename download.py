import os
import requests

from zipfile import ZipFile
from tqdm import tqdm

DIV2K_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K"
DATA_DIR = 'DIV2K'
TRAIN = "DIV2K_train_HR.zip"
VAL = "DIV2K_valid_HR.zip"

class Downloader:
    """ Downloader class used for downloading data from given url """

    units = {
        'KB' : 1024,
        'MB' : 1024 ** 2,
        'GB' : 1024 ** 3
    }

    def __init__(self, unit = 'KB'):
        assert unit in self.units.keys()
        
        self.unit = unit
        self.chunk_size = self.units[unit]

    def __call__(self, url, file):
        req = requests.get(url, stream=True)
        total_size = int(req.headers['content-length'])

        with open(file, 'wb') as f:
            for data in tqdm(iterable = req.iter_content(self.chunk_size), total = total_size / self.chunk_size, unit = self.unit):
                f.write(data)

    def download(self, url, file):
        self(url, file)
        

if __name__ == '__main__':

    # exit if images already downloaded
    if os.path.exists(DATA_DIR) and len(os.listdir(DATA_DIR)) != 0:
        exit(0)

    down = Downloader(unit='MB')
    
    print(f'Downloading: {TRAIN}')
    down(f'{DIV2K_URL}/{TRAIN}', TRAIN)

    print(F'Dowloading: {VAL}')
    down(f'{DIV2K_URL}/{VAL}', VAL)

    print('Unzipping')
    for type in [TRAIN, VAL]:
        with ZipFile(type, 'r') as zip:
            zip.extractall(DATA_DIR)



