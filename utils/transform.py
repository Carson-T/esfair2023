import albumentations
from albumentations import pytorch as AT

def transform(args):
    train_transforms = albumentations.Compose([
        albumentations.Resize(args["resize"], args["resize"]),
        albumentations.Transpose(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        albumentations.OneOf([
            albumentations.MotionBlur(blur_limit=5),
            albumentations.MedianBlur(blur_limit=5),
            albumentations.GaussianBlur(blur_limit=(1,5)),
            albumentations.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.5),

        albumentations.OneOf([
            albumentations.OpticalDistortion(distort_limit=1.0),
            albumentations.GridDistortion(num_steps=5, distort_limit=1.),
            albumentations.ElasticTransform(alpha=3),
        ], p=0.5),

        albumentations.CLAHE(clip_limit=0.5, p=0.5),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.5),
        albumentations.CoarseDropout(max_holes=1, max_height=int(args["resize"] * 0.1), max_width=int(args["resize"] * 0.1), min_height=1, min_width=1, p=0.5),
        albumentations.Normalize(),
        AT.ToTensorV2()
        ])

    val_transforms = albumentations.Compose([
        albumentations.Resize(args["resize"], args["resize"]),
        albumentations.Normalize(),
        AT.ToTensorV2()
    ])

    return train_transforms, val_transforms