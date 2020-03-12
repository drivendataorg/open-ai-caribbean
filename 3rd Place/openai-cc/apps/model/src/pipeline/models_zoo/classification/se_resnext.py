from .resnext import ResNeXt
import torch
import math

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import pretrainedmodels

__all__ = ['se_resnext50', 'se_resnext101', 'se_resnext101_64', 'se_resnext152']
model_urls = {'se_resnext50': 'https://nizhib.ai/share/pretrained/se_resnext50-5cc09937.pth'}


class SEBlock(nn.Module):
    def __init__(self, planes, reduction=16):
        super(SEBlock, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cSE = nn.Sequential(
            nn.Linear(planes, planes // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(planes // reduction, planes),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape

        cse = self.avg_pool(x).view(b, c)
        cse = self.cSE(cse).view(b, c, 1, 1)
        cse = x * cse

        return cse


class SEBottleneck(nn.Module):
    """
    SE-RexNeXt bottleneck type C
    """
    expansion = 4

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, downsample=None,
                 reduction=16):
        super(SEBottleneck, self).__init__()

        D = int(math.floor(planes * (baseWidth / 64)))
        C = cardinality

        self.conv1 = nn.Conv2d(inplanes, D * C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(D * C)
        self.conv2 = nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride,
                               padding=1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm2d(D * C)
        self.conv3 = nn.Conv2d(D * C, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(planes * 4, reduction)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def se_resnext50(num_classes=1000, pretrained=False):
    """Constructs a SE-ResNeXt-50 model."""
    model = ResNeXt(SEBottleneck, 4, 32, [3, 4, 6, 3], num_classes=num_classes)
    if pretrained:
        model.load_state_dict(torch.load('/home/kg2/.torch/models/se_resnext50-5cc09937.pth'))
        # model.load_state_dict(model_zoo.load_url(model_urls['se_resnext50']))
    return model


def se_resnext101(num_classes=1000):
    """Constructs a SE-ResNeXt-101 (32x4d) model."""
    model = ResNeXt(SEBottleneck, 4, 32, [3, 4, 23, 3], num_classes=num_classes)
    return model


def se_resnext101_64(num_classes=1000):
    """Constructs a SE-ResNeXt-101 (64x4d) model."""
    model = ResNeXt(SEBottleneck, 4, 64, [3, 4, 23, 3], num_classes=num_classes)
    return model


def se_resnext152(num_classes=1000):
    """Constructs a SE-ResNeXt-152 (32x4d) model."""
    model = ResNeXt(SEBottleneck, 4, 32, [3, 8, 36, 3], num_classes=num_classes)
    return model


def SEResNext50(pretrained, num_classes):
    model = se_resnext50(pretrained=pretrained)
    model.fc = nn.Linear(2048, num_classes, bias=True)
    return model


def SK_se_resnext50_32x4d(num_classes=1000, pretrained='imagenet'):
    model = pretrainedmodels.se_resnext50_32x4d()
    model.last_linear = nn.Linear(2048, num_classes, bias=True)
    return model


def SK_se_resnext101_32x4d(num_classes=1000, pretrained='imagenet'):
    model = pretrainedmodels.se_resnext101_32x4d()
    model.last_linear = nn.Linear(2048, num_classes, bias=True)
    return model
