import albumentations
from albumentations import pytorch as AT

def transform(args):
    train_transforms = albumentations.Compose([
        albumentations.Resize(args["resize"], args["resize"]),
        albumentations.Transpose(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        # albumentations.OneOf([
        #     albumentations.MotionBlur(blur_limit=7),
        #     albumentations.MedianBlur(blur_limit=7),
        #     albumentations.GaussianBlur(blur_limit=(3, 7)),
        #     albumentations.GaussNoise(var_limit=(10, 50)),
        # ], p=0.8),
        # albumentations.OneOf([
        #     albumentations.OpticalDistortion(distort_limit=2.0),       #光学畸变
        #     albumentations.GridDistortion(num_steps=5, distort_limit=1.),
        #     albumentations.ElasticTransform(alpha=3),  #弹性变换
        # ], p=0.8),
        # albumentations.CLAHE(clip_limit=4, p=0.5),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=45, border_mode=0, p=0.5),
        # albumentations.CoarseDropout(max_holes=1, max_height=int(args["resize"] * 0.3), max_width=int(args["resize"] * 0.3), min_height=1, min_width=1, p=0.5),
        albumentations.Normalize(),
        AT.ToTensorV2()
        ])

    val_transforms = albumentations.Compose([
        albumentations.Resize(args["resize"], args["resize"]),
        albumentations.Normalize(),
        AT.ToTensorV2()
    ])

    return train_transforms, val_transforms