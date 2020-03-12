import torch.nn as nn
import torch
from ..classification.resnext import resnext50, resnext101, resnext101_64, resnext152
from ..classification.se_resnext import se_resnext50, se_resnext101, se_resnext101_64, se_resnext152
from ..classification.se_resnext import SEBlock

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

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


class DecoderSEBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvRelu(in_channels, middle_channels),
            ConvRelu(middle_channels, out_channels))
        # SEBlock(planes=out_channels, reduction=16))

    def forward(self, x):
        return self.block(x)


class DecoderSEBlockV3(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvRelu(in_channels, middle_channels),
            ConvRelu(middle_channels, out_channels),
            SEBlock(planes=out_channels, reduction=16))

    def forward(self, x):
        return self.block(x)


class SENeXt50(nn.Module):
    def __init__(self, num_classes=1, num_filters=16, pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        encoder = se_resnext50(pretrained=pretrained)
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


class NeXt50(nn.Module):
    def __init__(self, num_classes=1, num_filters=16, pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        encoder = resnext50(pretrained=pretrained)
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


class NeXt101(nn.Module):
    def __init__(self, num_classes=1, num_filters=16, pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        encoder = resnext101(pretrained=pretrained)
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


class SENeXt50WithoutPooling(nn.Module):
    def __init__(self, num_classes=1, num_filters=16, pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        encoder = se_resnext50(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)

        self.encoder = nn.ModuleList([
            nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu),
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
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=3)

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


class MultiSENeXt50(nn.Module):
    def __init__(self, num_classes=1, num_filters=16, pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        encoder = se_resnext50(pretrained=pretrained)
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
            DecoderBlockV2(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)])

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(2048, 1)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        encoder_results = []
        for stage in self.encoder:
            x = stage(x)
            encoder_results.append(x.clone())
        x_cls = self.avgpool(x)
        x_cls = x_cls.view(x_cls.size(0), -1)
        x_cls = self.fc(x_cls).view(x_cls.size(0))

        x = self.center(self.pool(x))
        for i, decoder in enumerate(self.decoder):
            x = self.decoder[i](torch.cat([x, encoder_results[-i - 1]], 1))

        x = self.dec1(x)
        x = self.dec0(x)
        x = self.final(x)
        return x, x_cls


class MultiSESENeXt50(nn.Module):
    def __init__(self, num_classes=1, num_filters=16, pretrained=False):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        encoder = se_resnext50(pretrained=pretrained)
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

        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(2048, 1)

        self.center = DecoderSEBlockV2(2048, num_filters * 8 * 2, num_filters * 8)

        self.decoder = nn.ModuleList([
            DecoderSEBlockV2(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8),
            DecoderSEBlockV2(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8),
            DecoderSEBlockV2(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2),
            DecoderSEBlockV2(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2),
        ])

        self.dec1 = DecoderSEBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        encoder_results = []
        for stage in self.encoder:
            x = stage(x)
            encoder_results.append(x.clone())

        # x_cls = self.avgpool(x)
        # x_cls = x_cls.view(x_cls.size(0), -1)
        # x_cls = self.fc(x_cls).view(x_cls.size(0))
        # print(x_cls.shape)

        x = self.center(self.pool(x))
        # print(x.shape)

        for i, decoder in enumerate(self.decoder):
            x = self.decoder[i](torch.cat([x, encoder_results[-i - 1]], 1))
            # print(x.shape)

        x = self.dec1(x)
        # print(x.shape)
        x = self.dec0(x)
        # print(x.shape)
        x = self.final(x)
        # print(x.shape)
        # print('ok')

        return x  # , x_cls


class MultiSESENeXt50_2(nn.Module):
    def __init__(self, num_classes=1, num_filters=16, pretrained=False):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        encoder = se_resnext50(pretrained=pretrained)
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
        self.fc = nn.Linear(2048, 1)

        self.center = DecoderSEBlockV3(2048, num_filters * 8 * 2, num_filters * 8)

        self.decoder = nn.ModuleList([
            DecoderSEBlockV3(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8),
            DecoderSEBlockV3(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8),
            DecoderSEBlockV3(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2),
            DecoderSEBlockV3(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2),
        ])

        self.dec1 = DecoderSEBlockV3(num_filters * 2 * 2, num_filters * 2 * 2, num_filters)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        encoder_results = []
        for stage in self.encoder:
            x = stage(x)
            encoder_results.append(x.clone())

        # x_cls = self.avgpool(x)
        # x_cls = x_cls.view(x_cls.size(0), -1)
        # x_cls = self.fc(x_cls).view(x_cls.size(0))

        x = self.center(self.pool(x))

        for i, decoder in enumerate(self.decoder):
            x = self.decoder[i](torch.cat([x, encoder_results[-i - 1]], 1))

        x = self.dec1(x)
        x = self.dec0(x)
        x = self.final(x)

        return x  # , x_cls


class MultiSESENeXt101(nn.Module):
    def __init__(self, num_classes=1, num_filters=16, pretrained=False):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        encoder = se_resnext101(pretrained=pretrained)
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
        self.fc = nn.Linear(2048, 1)

        self.center = DecoderSEBlockV2(2048, num_filters * 8 * 2, num_filters * 8)

        self.decoder = nn.ModuleList([
            DecoderSEBlockV2(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8),
            DecoderSEBlockV2(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8),
            DecoderSEBlockV2(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2),
            DecoderSEBlockV2(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2),
        ])

        self.dec1 = DecoderSEBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        encoder_results = []
        for stage in self.encoder:
            x = stage(x)
            encoder_results.append(x.clone())

        # x_cls = self.avgpool(x)
        # x_cls = x_cls.view(x_cls.size(0), -1)
        # x_cls = self.fc(x_cls).view(x_cls.size(0))

        x = self.center(self.pool(x))

        for i, decoder in enumerate(self.decoder):
            x = self.decoder[i](torch.cat([x, encoder_results[-i - 1]], 1))

        x = self.dec1(x)
        x = self.dec0(x)
        x = self.final(x)

        return x
