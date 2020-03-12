from pretrainedmodels import inceptionv3, inceptionresnetv2, inceptionv4
import torch.nn as nn


def InceptionV3(num_classes, pretrained=False):
    if pretrained:
        model = inceptionv3(pretrained='imagenet')
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Linear(2048, num_classes)
    return model


def InceptionResNetV2(num_classes, pretrained=False):
    if pretrained:
        model = inceptionresnetv2(pretrained='imagenet')
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Linear(1536, num_classes)
    return model

