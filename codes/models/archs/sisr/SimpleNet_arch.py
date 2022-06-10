import functools
import torch.nn as nn
import models.archs.arch_util as arch_util
import math

class SimpleNet(nn.Module):
    def __init__(self, nf=64, RBs=10, upscale=2):
        super(SimpleNet, self).__init__()
        self.nf = nf

        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

        self.conv1 = nn.Conv2d(2, nf, 3, 1, 1, bias=True)
        self.res_blocks = arch_util.make_layer(ResidualBlock_noBN_f, RBs)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.upsample = Upsample(upscale, nf)
        self.conv_last = nn.Conv2d(nf, 1, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        B, C, H, W = x.size()

        x = self.lrelu(self.conv1(x))
        x = self.res_blocks(x)
        x = self.lrelu(self.conv2(x))
        x = self.conv_last(self.upsample(x))

        return x


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)