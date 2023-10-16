import os
import requests
import shutil

from zipfile import ZipFile
from tqdm import tqdm

DIV2K_URL = 'http://data.vision.ee.ethz.ch/cvl/DIV2K'
DATA, TRAIN, VALID = 'DIV2K', 'DIV2K_train_HR', 'DIV2K_valid_HR'

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
    if os.path.exists(DATA) and len(os.listdir(DATA)) != 0:
        exit(0)

    down = Downloader(unit='MB')
    os.mkdir(DATA)

    for dir in [TRAIN, VALID]:
        
        print(f'[{dir}]: downloading')
        down(f'{DIV2K_URL}/{dir}.zip', f'{dir}.zip')
        
        print(f'[{dir}]: unzipping')
        with ZipFile(f'{dir}.zip', 'r') as zip:
            zip.extractall()

        print(f'[{dir}]: removing zip file')
        for f in os.listdir(dir):
            shutil.move(f'{dir}/{f}', f'{DATA}/{f}')

        os.rmdir(dir)
        os.remove(f'{dir}.zip')

    print(f'total img count: {len(os.listdir(DATA))}')
