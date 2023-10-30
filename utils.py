""" Superres model related utilities """

import random

import torchvision.transforms.functional as FT
from PIL import Image

INPUT_CONVERSIONS = {
    'pil' : FT.to_tensor,
    '[0, 255]': lambda img: img / 255,
    '[0, 1]' : lambda img: img,
    '[-1, 1]': lambda img: (img + 1) / 2
}

OUTPUT_CONVERSIONS = {
    'pil' : FT.to_pil_image,
    '[0, 255]': lambda img: img * 255,
    '[0, 1]': lambda img: img,
    '[-1, 1]': lambda img: img * 2 - 1
}


def convert_img(img, input_format, output_format):
    """ Convert given image from source format onto target format

    Args:
    img -- image to convert
    input_format -- source image format ('pil', '[0, 255]', '[0, 1]', '[-1, 1]')
    output_format -- target image format ('pil', '[0, 255]', '[0, 1]', '[-1, 1]')
    """
    assert input_format in INPUT_CONVERSIONS
    assert output_format in OUTPUT_CONVERSIONS

    return OUTPUT_CONVERSIONS[output_format](INPUT_CONVERSIONS[input_format](img))


class ImgTransformer:
    """ Img transformer utility class for cropping and scaling images
    
    Args:
    lr_output -- low resolution image output format
    hr_output -- high resolution image output format
    crop -- training image crop size (default: 64)
    scale -- low resolution image scale (default: 2)
    is_train -- choose if images should be for training or validation (default: True)
    """

    def __init__(self, lr_output, hr_output, crop=64, scale=2, is_train=True):
        self.crop = crop
        self.scale = scale
        self.is_train = is_train
        self.lr_output = lr_output
        self.hr_output = hr_output

        assert crop % scale == 0

    def __call__(self, img):
        """ Transform given image and return pair of low resultion image and original image

        Args:
        img -- source image to transform 
        """
        if self.is_train:
            # crop random cut from given image based on specified scrop dimensions
            assert self.crop <= min(img.size)

            # crop HR image
            left, top = random.randint(0, img.width - self.crop), random.randint(0, img.height - self.crop)
            right, bottom = left + self.crop, top + self.crop

            hr_img = img.crop((left, top, right, bottom))
        else:
            # crop biggest possible image part divisible by scale
            right, bottom = (img.width // self.scale) * self.scale, (img.height // self.scale) * self.scale
            hr_img = img.crop((0, 0, right, bottom))

        # downscale hr image
        lr_img = hr_img.resize((hr_img.width // self.scale, hr_img.height // self.scale), Image.BICUBIC)

        assert lr_img.width * self.scale == hr_img.width
        assert lr_img.height * self.scale == hr_img.height
        assert (hr_img.width % self.scale, hr_img.height % self.scale) == (0, 0)

        # return lres and hres images in specified output type
        return convert_img(lr_img, 'pil', self.lr_output), convert_img(hr_img, 'pil', self.hr_output)
