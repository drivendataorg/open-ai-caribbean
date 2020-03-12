from pretrainedmodels import pnasnet5large
import torch.nn as nn


def PNasNet5Large(pretrained, num_classes):
    if pretrained:
        model = pnasnet5large(pretrained='imagenet+background')
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Linear(4320, num_classes)
    return model
