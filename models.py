""" NN modules file with model components """

from math import log2
from torch import nn


class SubPixelBlock(nn.Module):
    """Subpixel conv block used for img upscaling using combined channels"""

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
    """Constant dimension conv block with optional normalization and activation function"""

    ACTIVATIONS = {
        None: None,
        "prelu": nn.PReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
    }

    def __init__(self, in_channels, out_channels, k, norm=False, activation=None):
        super().__init__()

        assert activation in self.ACTIVATIONS

        # main conv block
        layers = [nn.Conv2d(in_channels, out_channels, k, padding=k // 2)]

        # optional conv layers
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation is not None:
            layers.append(self.ACTIVATIONS[activation.lower()])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class ResidualConvBlock(nn.Module):
    """Residual conv block with 2 conv layers and residual connection"""

    def __init__(self, n_channels, k, norm=False):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            ConvBlock(n_channels, n_channels, k, norm, "prelu"),
            ConvBlock(n_channels, n_channels, k, norm, None),
        )

    def forward(self, x):
        skip, x = x, self.conv_blocks(x)
        return x + skip


class SResNet(nn.Module):
    """Super resolution upscaling model"""

    def __init__(self, in_channels, out_channels, res_block_cnt=4, scale=4, norm=False):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 64, 9, norm, "prelu")

        self.res_blocks = nn.Sequential(
            *[ResidualConvBlock(64, 3, norm) for _ in range(res_block_cnt)]
        )

        self.conv2 = ConvBlock(64, 64, 9, norm)

        self.subpix_blocks = nn.Sequential(
            *[SubPixelBlock(64, 3, scale=2) for _ in range(int(log2(scale)))]
        )

        self.conv3 = ConvBlock(64, out_channels, 9, norm, "sigmoid")

    def forward(self, x):
        # conv1
        x = self.conv1(x)

        # res blocks with conv2
        skip, x = x, self.res_blocks(x)
        x = self.conv2(x)
        x += skip

        # subpix blocks
        x = self.subpix_blocks(x)

        # conv3 with sigmoid activation
        x = self.conv3(x)

        return x
