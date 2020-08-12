""" 
Source: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py (Edited)
"""

import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

""" Parts of the U-Net model """


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential()

        self.double_conv.add_module('conv1', nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1))
        if bn: self.double_conv.add_module('bn1', nn.BatchNorm3d(mid_channels))
        self.double_conv.add_module('relu1', nn.ReLU(inplace=True))
        self.double_conv.add_module('conv2', nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1))
        if bn: self.double_conv.add_module('bn2', nn.BatchNorm3d(out_channels))
        self.double_conv.add_module('relu2', nn.ReLU(inplace=True))
    

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, bn=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels, bn=bn)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, bn=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, bn=bn)
        else:
            self.up = nn.ConvTranspose3d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, bn=bn)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


""" Full assembly of the parts to form the complete network  """


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, bn=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64, bn=bn)
        self.down1 = Down(64, 128, bn=bn)
        self.down2 = Down(128, 256, bn=bn)
        self.up2 = Up(256 + 128, 128, bilinear, bn=bn)
        self.up1 = Up(128 + 64, 64, bilinear, bn=bn)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up2(x3, x2)
        x = self.up1(x, x1)

        # logits
        logits = self.outc(x)

        # output
        y = F.softmax(logits, dim=1)

        # projections
        yi, yi_idx = torch.max(y, 2)
        yj, yj_idx = torch.max(y, 3)
        yk, yk_idx = torch.max(y, 4)

        return y, yi, yj, yk
