""" Script for downloading DIV2K dataset images for superres model training """

import os
import sys
import shutil
from zipfile import ZipFile
import requests

from tqdm import tqdm

DIV2K_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K"
DATA, TRAIN, VALID = "DIV2K", "DIV2K_train_HR", "DIV2K_valid_HR"


class Downloader:
    """ Downloader class used for downloading data from given url
    
    Args:
    unit -- data download unit (default: KB)
    """

    units = {"KB": 1024, "MB": 1024**2, "GB": 1024**3}

    def __init__(self, unit="KB"):
        assert unit in self.units

        self.unit = unit
        self.chunk_size = self.units[unit]

    def __call__(self, url, filename):
        """ Download data from given url 
        
        Args:
        url -- web url containing data
        filename -- downloaded data name
        """
        try:
            req = requests.get(url, stream=True, timeout=20)
        except Exception as exc:
            raise exc

        total_size = int(req.headers["content-length"])

        with open(filename, "wb") as file:
            for data in tqdm(
                iterable=req.iter_content(self.chunk_size),
                total=total_size / self.chunk_size,
                unit=self.unit,
            ):
                file.write(data)


if __name__ == "__main__":
    # exit if images already downloaded
    if os.path.exists(DATA) and len(os.listdir(DATA)) != 0:
        print(f"{DATA} folder already exists and it is not empty")
        sys.exit(0)

    down = Downloader(unit="MB")
    os.mkdir(DATA)

    for data_dir in [TRAIN, VALID]:
        try:
            print(f"[{data_dir}]: downloading")
            down(f"{DIV2K_URL}/{data_dir}.zip", f"{data_dir}.zip")
        except requests.Timeout:
            print("Request time out")
            os.rmdir(DATA)
            sys.exit(2)

        print(f"[{data_dir}]: unzipping")
        with ZipFile(f"{data_dir}.zip", "r") as zipfile:
            zipfile.extractall()

        print(f"[{data_dir}]: removing zip file")
        for f in os.listdir(data_dir):
            shutil.move(f"{data_dir}/{f}", f"{DATA}/{f}")

        os.rmdir(data_dir)
        os.remove(f"{data_dir}.zip")

    print(f"total img count: {len(os.listdir(DATA))}")
