""" Superres model related utilities """

import random
import torch
import numpy as np

from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


class ImgTransform:
    """Img transformer utility class for cropping and scaling images"""

    def __init__(self, crop=128, scale=4, is_train=True, output="Tensor"):
        self.crop = crop
        self.scale = scale
        self.is_train = is_train
        self.output = output

        assert crop % scale == 0
        assert output in ["Tensor", "Numpy", "PIL"]

    def __call__(self, img):
        if self.is_train:
            # crop random cut from given image based on specified scrop dimensions
            assert self.crop <= min(img.size)

            # crop HR image
            left, top = random.randint(0, img.width - self.crop), random.randint(
                0, img.height - self.crop
            )
            right, bottom = left + self.crop, top + self.crop

            hr_img = img.crop((left, top, right, bottom))
        else:
            # crop biggest possible image part divisible by scale
            right, bottom = (img.width // self.scale) * self.scale, (
                img.height // self.scale
            ) * self.scale
            hr_img = img.crop((0, 0, right, bottom))

        # downscale hr image
        lr_img = hr_img.resize(
            (hr_img.width // self.scale, hr_img.height // self.scale), Image.BICUBIC
        )

        assert lr_img.width * self.scale == hr_img.width
        assert lr_img.height * self.scale == hr_img.height
        assert (hr_img.width % self.scale, hr_img.height % self.scale) == (0, 0)

        # return lres and hres images in specified output type
        if self.output == "Tensor":
            return pil_to_tensor(lr_img).type(torch.float), pil_to_tensor(hr_img).type(
                torch.float
            )

        if self.output == "Numpy":
            return np.array(lr_img), np.array(hr_img)

        return lr_img, hr_img
