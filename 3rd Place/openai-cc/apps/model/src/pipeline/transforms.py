import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor


def pre_transform(resize):
    transforms = [
        A.Resize(resize, resize),
    ]
    return A.Compose(transforms)


def post_transform():
    return A.Compose([
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.22)),
        ToTensor()])


def mix_transform(resize):
    return A.Compose([
        pre_transform(resize=resize),
        A.GridDistortion(),
        A.RandomRotate90(always_apply = True),
        A.Rotate(limit=10, border_mode=0, p=0.5),
#         A.OneOf([
        A.Flip(p=0.5),
        A.Transpose(p=0.5),
#         ], p=0.5),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.3),
        A.OneOf([
            A.Blur(blur_limit=3, p=0.2),
            A.MotionBlur(blur_limit=5, p=0.4),
            A.MedianBlur(blur_limit=5, p=0.4)
        ], p=0.3),
        A.GaussNoise((10, 50), p=0.1),
        post_transform()
    ])


def test_transform(resize):
    return A.Compose([
        A.Resize(resize, resize),
        post_transform()]
    )
