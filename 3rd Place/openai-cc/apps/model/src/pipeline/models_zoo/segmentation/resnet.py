import torch.nn as nn
import math
import torchvision
from torch.nn import functional as F

import torch
from ..classification.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .resnext import DecoderSEBlockV2


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(ConvBn2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        # self.bn = SynchronizedBatchNorm2d(out_channels)

    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = ConvBn2d(in_channels, channels, kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.upsample(x, scale_factor=2, mode='bilinear')
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        return x


class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True))
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels))

    def forward(self, x):
        return self.block(x)


class MultiResnet50(nn.Module):
    def __init__(self, num_classes=1, num_filters=16, pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        encoder = resnet50(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)

        self.encoder = nn.ModuleList([
            nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu,
                self.pool),
            encoder.layer1,
            encoder.layer2,
            encoder.layer3,
            encoder.layer4])

        self.center = DecoderBlockV2(2048, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.decoder = nn.ModuleList([
            DecoderBlockV2(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv),
            DecoderBlockV2(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv),
            DecoderBlockV2(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv),
            DecoderBlockV2(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv),
        ])

        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        encoder_results = []
        for stage in self.encoder:
            x = stage(x)
            encoder_results.append(x.clone())

        x = self.center(self.pool(x))

        for i, decoder in enumerate(self.decoder):
            x = self.decoder[i](torch.cat([x, encoder_results[-i - 1]], 1))

        x = self.dec1(x)
        x = self.dec0(x)
        x = self.final(x)

        return x


class MultiResnet34(nn.Module):
    def __init__(self, num_classes=1, num_filters=16, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        encoder = resnet34(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)

        self.encoder = nn.ModuleList([
            nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu,
                self.pool),
            encoder.layer1,
            encoder.layer2,
            encoder.layer3,
            encoder.layer4])

        self.avgpool = nn.AvgPool2d(3)
        self.fc = nn.Linear(512, 1)

        self.center = DecoderSEBlockV2(512, num_filters * 8 * 2, num_filters * 8)

        self.decoder = nn.ModuleList([
            DecoderSEBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8),
            DecoderSEBlockV2(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8),
            DecoderSEBlockV2(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2),
            DecoderSEBlockV2(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2),
        ])

        self.dec1 = DecoderSEBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        encoder_results = []
        for stage in self.encoder:
            x = stage(x)
            encoder_results.append(x.clone())

        x = self.center(self.pool(x))

        for i, decoder in enumerate(self.decoder):
            x = self.decoder[i](torch.cat([x, encoder_results[-i - 1]], 1))

        x = self.dec1(x)
        x = self.dec0(x)
        x = self.final(x)

        return x


class MultiResnet18(nn.Module):
    def __init__(self, num_classes=1, num_filters=16, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        encoder = resnet18(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)

        self.encoder = nn.ModuleList([
            nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu,
                self.pool),
            encoder.layer1,
            encoder.layer2,
            encoder.layer3,
            encoder.layer4])

        self.avgpool = nn.AvgPool2d(3)
        self.fc = nn.Linear(512, 1)

        self.center = DecoderSEBlockV2(512, num_filters * 8 * 2, num_filters * 8)

        self.decoder = nn.ModuleList([
            DecoderSEBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8),
            DecoderSEBlockV2(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8),
            DecoderSEBlockV2(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2),
            DecoderSEBlockV2(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2),
        ])

        self.dec1 = DecoderSEBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        encoder_results = []
        for stage in self.encoder:
            x = stage(x)
            encoder_results.append(x.clone())

        x = self.center(self.pool(x))

        for i, decoder in enumerate(self.decoder):
            x = self.decoder[i](torch.cat([x, encoder_results[-i - 1]], 1))

        x = self.dec1(x)
        x = self.dec0(x)
        x = self.final(x)

        return x