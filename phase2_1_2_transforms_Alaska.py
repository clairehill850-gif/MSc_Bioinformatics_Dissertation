#!/usr/bin/env python3
# What this does: defines how training images are altered ALASKA
import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BORDER = cv2.BORDER_REFLECT_101


def _positional_jitter(limit=0.05, p=0.5):
    import warnings
    base = dict(translate_percent={"x": (-limit, limit), "y": (-limit, limit)},
                scale=1.0, rotate=0, p=p)
    for border_kw in ("border_mode", "mode"):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", message="Argument.*not valid")
                return A.Affine(**base, **{border_kw: BORDER})
        except Exception:
            continue
    return A.Affine(**base)


def get_train_transform(size=224):
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=180, p=1.0, border_mode=BORDER),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15,
                             val_shift_limit=10, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
        _positional_jitter(limit=0.05, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transform(size=224):
    if os.environ.get('POLLEN_GRAY') == '1':
        return get_val_transform_gray(size)
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# Setting recorded to W&B
AUG_CONFIG = {
    "rotate_limit": 180, "hflip": 0.5, "vflip": 0.5,
    "brightness_contrast": 0.2, "hue_sat_value": [10, 15, 10],
    "color_jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2,
                     "hue": 0.05, "p": 0.8},
    "shift_jitter": {"shift_limit": 0.05, "p": 0.5},
    "elastic": {"alpha": 1, "sigma": 50, "p": 0.3},
    "normalize": "imagenet", "border": "reflect_101",
}



# Shrinking the image at random and blurring it (ALASKA)

def _downscale(p=0.5):
    import warnings
    for kwargs in ({"scale_range": (0.3, 0.7), "interpolation_pair": {"downscale": cv2.INTER_AREA, "upscale": cv2.INTER_LANCZOS4}},
                   {"scale_min": 0.3, "scale_max": 0.7}):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                return A.Downscale(p=p, **kwargs)
        except Exception:
            continue
    print("warning: A.Downscale signature not recognised - using GaussianBlur only")
    return A.GaussianBlur(blur_limit=(3, 7), p=p)


def _softness(p=0.4):
    try:
        return A.OneOf([A.GaussianBlur(blur_limit=(3, 7)),
                        A.Defocus(radius=(2, 5))], p=p)
    except Exception:
        return A.GaussianBlur(blur_limit=(3, 7), p=p)


def get_train_transform_domain(size=224):
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=180, p=1.0, border_mode=BORDER),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15,
                             val_shift_limit=10, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
        _positional_jitter(limit=0.05, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        _downscale(p=0.5),
        _softness(p=0.4),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


AUG_CONFIG_DOMAIN = dict(AUG_CONFIG)
AUG_CONFIG_DOMAIN.update({
    "downscale": {"scale_range": [0.3, 0.7], "p": 0.5},
    "softness": {"gaussian_blur": [3, 7], "defocus_radius": [2, 5], "p": 0.4},
})


#Greyscale
def get_train_transform_gray(size=224):
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=180, p=1.0, border_mode=BORDER),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        _positional_jitter(limit=0.05, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        _downscale(p=0.5),
        _softness(p=0.4),
        A.ToGray(p=1.0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# Validation greyscale
def get_val_transform_gray(size=224):
    return A.Compose([
        A.ToGray(p=1.0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
