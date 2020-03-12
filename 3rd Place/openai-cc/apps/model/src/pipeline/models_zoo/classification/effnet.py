from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import get_same_padding_conv2d, round_filters


def EffNet(num_classes=1000, model_name='efficientnet-b0', in_channels=3):
    model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)

    # Conv2d = get_same_padding_conv2d(image_size=model._global_params.image_size)
    # out_channels = round_filters(32, model._global_params)
    # model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)

    return model
