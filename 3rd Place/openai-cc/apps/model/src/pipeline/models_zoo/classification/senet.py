from pretrainedmodels import senet154
import torch.nn as nn


def SENet154(pretrained, num_classes):
    if pretrained:
        model = senet154(pretrained='imagenet')
    else:
        model = senet154()

    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Linear(2048, num_classes, bias=True)

    return model
