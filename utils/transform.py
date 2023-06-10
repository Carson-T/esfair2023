import albumentations
from albumentations import pytorch as AT

def transform(args):
    train_transforms = albumentations.Compose([
            albumentations.Transpose(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.RandomBrightness(limit=0.2, p=0.75),
            albumentations.RandomContrast(limit=0.2, p=0.75),
            albumentations.OneOf([
                albumentations.MotionBlur(blur_limit=5),
                albumentations.MedianBlur(blur_limit=5),
                albumentations.GaussianBlur(blur_limit=5),
                albumentations.GaussNoise(var_limit=(5.0, 30.0)),
            ], p=0.7),

            albumentations.OneOf([
                albumentations.OpticalDistortion(distort_limit=1.0),
                albumentations.GridDistortion(num_steps=5, distort_limit=1.),
                albumentations.ElasticTransform(alpha=3),
            ], p=0.7),

            albumentations.CLAHE(clip_limit=4.0, p=0.7),
            albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.5),
            albumentations.Resize(args["resize"], args["resize"]),
            albumentations.Cutout(max_h_size=int(args["resize"] * 0.2), max_w_size=int(args["resize"] * 0.2), num_holes=1, p=0.5),
            albumentations.Normalize(),
            AT.ToTensorV2()
        ])

    val_transforms = albumentations.Compose([
        albumentations.Resize(args["resize"], args["resize"]),
        albumentations.Normalize(),
        AT.ToTensorV2()
    ])

    return train_transforms, val_transforms