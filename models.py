""" Neural network building blocks with sresnet model """

from math import log2
from torch import nn

import torch


class SubPixelBlock(nn.Module):
    """ Subpixel conv block used for img upscaling using combined channels
    
    Args:
    n_channels -- number of input channels
    k -- kernel size
    scale -- block scaling factor (default: 2)
    """

    def __init__(self, n_channels, k, scale=4):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(n_channels, n_channels * (scale**2), k, padding=k // 2),
            nn.PixelShuffle(scale),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class ConvBlock(nn.Module):
    """ Constant dimension conv block with optional normalization and activation function
    
    Args:
    in_channels -- number of input channels
    out_channels -- number of output channels
    k -- kernel size
    norm -- True / False param controlling batch normalization (default: False)
    activation -- convblock output activation function (default: None)
    """

    ACTIVATIONS = {
        None: None,
        "relu" : nn.ReLU(),
        "prelu": nn.PReLU(),
        "lrelu": nn.LeakyReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
    }

    def __init__(self, in_channels, out_channels, k, stride=1, norm=False, activation=None):
        super().__init__()

        assert activation in self.ACTIVATIONS

        # main conv block

        layers = [nn.Conv2d(in_channels, out_channels, k, stride=stride, padding=k // 2)]

        # optional conv layers
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation is not None:
            layers.append(self.ACTIVATIONS[activation.lower()])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class ResidualConvBlock(nn.Module):
    """ Residual conv block with 2 conv layers and residual connection
    
    Args:
    n_channels -- input and output channels
    k -- kernel size
    norm -- True / False param controlling batch normalization (default: False)
    """

    def __init__(self, n_channels, k, stride=1, norm=False):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            ConvBlock(n_channels, n_channels, k, stride, norm, 'prelu'),
            ConvBlock(n_channels, n_channels, k, stride, norm, None),
        )

    def forward(self, x):
        skip, x = x, self.conv_blocks(x)
        return x + skip


class SResNet(nn.Module):
    """ Super resolution upscaling model
    
    Args:
    in_channels -- number of input channels (default: 3)
    out_channels -- number of output channels (default: 3)
    n_channels -- number of inner convolution blocks channels (default: 64)
    small_kernel -- small kernel size used in residual blocks and subpixel blocks (default: 3)
    large_kernel -- large kernel size used in normal convolution blocks (default: 9)
    res_block_cnt -- number of chained residual blocks (default: 10)
    scale -- model scale factor (default: 2)
    norm -- True / False param controlling batch normalization (default: False)
    """

    def __init__(self, in_channels=3, out_channels=3, n_channels=64, 
                 small_kernel=3, large_kernel=9, res_block_cnt=10, 
                 scale=2, norm=False
    ):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, n_channels, large_kernel, 1, norm, 'prelu')

        self.res_blocks = nn.Sequential(
            *[ResidualConvBlock(n_channels, small_kernel, 1, norm) for _ in range(res_block_cnt)]
        )

        self.conv2 = ConvBlock(n_channels, n_channels, large_kernel, 1, norm)

        self.subpix_blocks = nn.Sequential(
            *[SubPixelBlock(n_channels, small_kernel, scale=2) for _ in range(int(log2(scale)))]
        )

        self.conv3 = ConvBlock(n_channels, out_channels, large_kernel, 1, norm, 'tanh')

    def forward(self, x):
        # conv1
        x = self.conv1(x)

        # res blocks with conv2
        skip, x = x, self.res_blocks(x)
        x = self.conv2(x)
        x += skip

        # subpix blocks
        x = self.subpix_blocks(x)

        # conv3 with tanh activation
        x = self.conv3(x)

        return x

class Generator(SResNet):
    def __init__(self, in_channels=3, out_channels=3, n_channels=64, 
                 small_kernel=3, large_kernel=9, res_block_cnt=10, 
                 scale=2, norm=False
    ):
        super().__init__(in_channels=in_channels, out_channels=out_channels, n_channels=n_channels, 
                 small_kernel=small_kernel, large_kernel=large_kernel, res_block_cnt=res_block_cnt, 
                 scale=scale, norm=norm)

    
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, norm=False):
        super().__init__()

        self.conv = nn.Sequential(*[
            ConvBlock(in_channels, 64, 3, stride=1, norm=False, activation='lrelu'),

            ConvBlock(64, 64, 3, stride=2, norm=norm, activation='lrelu'),

            ConvBlock(64, 128, 3, stride=1, norm=norm, activation='lrelu'),
            ConvBlock(128, 128, 3, stride=2, norm=norm, activation='lrelu'),

            ConvBlock(128, 256, 3, stride=1, norm=norm, activation='lrelu'),
            ConvBlock(256, 256, 3, stride=2, norm=norm, activation='lrelu'),

            ConvBlock(256, 512, 3, stride=1, norm=norm, activation='lrelu'),
            ConvBlock(512, 512, 3, stride=2, norm=norm, activation='lrelu'),

            # this should enable different image sizes
            nn.AdaptiveAvgPool2d((6, 6))
        ])

        self.dense = nn.Sequential(*[
            nn.Linear(36 * 512, 1024),
            nn.PReLU(),

            nn.Linear(1024, 1)
        ])

    def forward(self, x):
        batch_size = x.size(0)

        # run trough convolutional layers
        x = self.conv(x)

        # run flattened vector trough dense layers
        return self.dense(x.view(batch_size, -1))
