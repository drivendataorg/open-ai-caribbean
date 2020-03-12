from torchvision.models import densenet121, densenet161
import torch.nn as nn


def DenseNet121(pretrained, num_classes):
    model = densenet121(pretrained=pretrained)

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(1024, num_classes, bias=True)
    )
    return model


def DenseNet161(pretrained, num_classes):
    model = densenet161(pretrained=pretrained)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(2208, num_classes, bias=True)
    )
    return model
