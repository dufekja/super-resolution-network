from torch import nn

class SubPixelBlock(nn.Module):
    def __init__(self, scale=4, k=3, n_channels=64):
        super().__init__()
        self.scale = scale
        self.k = k
        self.n_channels = n_channels

        self.layers = nn.Sequential(
            nn.Conv2d(self.n_channels, self.n_channels * (self.scale ** 2), self.k, padding=self.k // 2),
            nn.PixelShuffle(self.scale),
            nn.PReLU()
        )    
    
    def forward(self, x):
        return self.layers(x)

class ConvBlock(nn.Module):    
    def __init__(self, in_channels, out_channels, k=3, norm=False, activation=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.norm = norm

        # insert conv layer
        self.layers = [nn.Conv2d(self.in_channels, self.out_channels, self.k, padding=self.k // 2)]

        # insert batch norm layer
        if norm: self.layers.append(nn.BatchNorm2d())

        # insert activation func
        if activation is not None:
            self.layers.append(
                {
                    'prelu' : nn.PReLU(),
                    'tanh' : nn.Tanh()
                }[activation.lower()]
            )
        
        self.conv = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.conv(x)
        

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, norm=False, activation=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.norm = norm
        self.activation = activation

        self.conv_blocks = nn.Sequential(
            ConvBlock(self.in_channels, self.out_channels, self.k, self.norm, self.activation),
            ConvBlock(self.in_channels, self.out_channels, self.k, self.norm)
        )
        
    def forward(self, x):
        residual, x = x, self.conv_blocks(x)
        return x + residual

class SResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, scale=4, residual_cnt=1, sub_pix_cnt=1, norm=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale
        self.residual_cnt = residual_cnt
        self.sub_pix_cnt = sub_pix_cnt
        self.norm = norm

        self.conv1 = ConvBlock(self.in_channels, 64, 9, activation='prelu')

        self.residual_blocks = nn.Sequential(
            *[ResidualConvBlock(64, 64, 3, activation='prelu') for _ in range(self.residual_cnt)]
        )

        self.conv2 = ConvBlock(64, 64, 9)

        self.sub_pixel_blocks = nn.Sequential(
            *[SubPixelBlock(self.scale) for _ in range(self.sub_pix_cnt)]
        )

        self.conv3 = ConvBlock(64, self.out_channels, 9, activation='tanh')

    def forward(self, x):

        # first conv with prelu
        x = self.conv1(x)

        # residual blocks with skipped conn
        skip, x = x, self.residual_blocks(x)
        x = self.conv2(x)
        x += skip

        # subpix blocks
        x = self.sub_pixel_blocks(x)
        
        # tanh conv block
        x = self.conv3(x)

        return x