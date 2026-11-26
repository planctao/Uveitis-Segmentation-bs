# Vendored from cruelMouth/PBE-UNet/network/PBEUNet.py. See provenance.json.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import log


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class CMUNeXtBlock(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, k=3):
        super(CMUNeXtBlock, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    # deep wise
                    nn.Conv2d(ch_in, ch_in, kernel_size=(k, k), groups=ch_in, padding=(k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
                nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in * 4),
                nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in)
            ) for i in range(depth)]
        )
        self.up = conv_block(ch_in, ch_out)

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        return x


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class BoundaryDetection(nn.Module):
    def __init__(self, ch_in):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_in // 2),
            nn.ReLU(),
            nn.Conv2d(ch_in // 2, 1, kernel_size=1)  # 输出边界概率图
        )

    def forward(self, x):
        return self.conv(x)


class BAFM(nn.Module):
    def __init__(self, main_ch):
        super().__init__()
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(main_ch + 1, main_ch, 3, padding=1),  # 输入通道=主干通道+边界通道
            nn.BatchNorm2d(main_ch),
            nn.ReLU(inplace=True)
        )

        self.boundary_propagate = nn.Sequential(
            nn.Conv2d(1, main_ch // 4, 3, padding=1),
            nn.BatchNorm2d(main_ch // 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(main_ch // 4, main_ch // 4, 3, groups=main_ch // 4, padding=1),
            nn.Conv2d(main_ch // 4, main_ch // 2, 1),
            nn.BatchNorm2d(main_ch // 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(main_ch // 2, main_ch // 2, 5, groups=main_ch // 2, padding=2),
            nn.Conv2d(main_ch // 2, main_ch, 1),

        )

        self.cbr = ConvBNR(main_ch, main_ch,3)


    def forward(self, x, boundary):
        fused = self.conv_fusion(torch.cat([x, boundary], dim=1))
        attention = self.boundary_propagate(boundary)
        refined = fused * attention + self.cbr(x)

        return refined


class ConvBNR(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, dilation=1,bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, scaling=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // scaling, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // scaling, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        return x


class CBAM_Attention(nn.Module):
    def __init__(self, channel, scaling=16, kernel_size=3):
        super(CBAM_Attention, self).__init__()
        self.conv2d = ConvBNR(channel, channel, 3)
        self.channelattention = ChannelAttention(channel, scaling=scaling)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.conv2d(x)
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class ECA(nn.Module):
    def __init__(self, channel):
        super(ECA, self).__init__()
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d = ConvBNR(channel, channel, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv2d(x)
        wei = self.avg_pool(x)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        x = x * wei

        return x


class SAAM(nn.Module):
    def __init__(self, hchannel, channel):
        super(SAAM, self).__init__()
        self.conv1_1 = Conv1x1(hchannel, channel)

        self.conv3_1 = ConvBNR(channel // 4, channel // 4, 3)
        self.dconv5_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=2)
        self.dconv7_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=3)
        self.dconv9_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=4)

        self.conv1_2 = Conv1x1(channel, channel)
        self.eca = ECA(channel)
        self.conv3_3 = ConvBNR(channel, channel, 3)



    def forward(self, x):
        x = self.conv1_1(x)
        xc = torch.chunk(x, 4, dim=1)

        x0 = self.conv3_1(xc[0] + xc[1])
        x1 = self.dconv5_1(xc[1] + x0 + xc[2])
        x2 = self.dconv7_1(xc[2] + x1 + xc[3])
        x3 = self.dconv9_1(xc[3] + x2)

        xx = self.conv1_2(torch.cat((x0, x1, x2, x3), dim=1))

        xx = self.eca(xx)
        x = self.conv3_3(x + xx)

        return x


class PBEUNet(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super(PBEUNet, self).__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

        self.boundary5 = BoundaryDetection(160)
        self.boundary4 = BoundaryDetection(128)
        self.boundary3 = BoundaryDetection(32)
        self.boundary2 = BoundaryDetection(16)

        self.bafm5 = BAFM(160)
        self.bafm4 = BAFM(128)
        self.bafm3 = BAFM(32)
        self.bafm2 = BAFM(16)

        self.cbam5 = CBAM_Attention(160)
        self.cbam4 = CBAM_Attention(128)
        self.cbam3 = CBAM_Attention(32)
        self.cbam2 = CBAM_Attention(16)

        self.saam5 = SAAM(160 * 2, 160)
        self.saam4 = SAAM(128 * 2, 128)
        self.saam3 = SAAM(32 * 2, 32)
        self.saam2 = SAAM(16 * 2, 16)


    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.saam5(d5)
        b5 = self.boundary5(d5)
        d5 = self.bafm5(d5, b5)
        d5 = self.cbam5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.saam4(d4)
        b4 = self.boundary4(d4)
        d4 = self.bafm4(d4, b4)
        d4 = self.cbam4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.saam3(d3)
        b3 = self.boundary3(d3)
        d3 = self.bafm3(d3, b3)
        d3 = self.cbam3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.saam2(d2)
        b2 = self.boundary2(d2)
        d2 = self.bafm2(d2, b2)
        d2 = self.cbam2(d2)

        d1 = self.Conv_1x1(d2)

        return d1, [b5, b4, b3, b2]

