import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Literal

def get_transforms(image_height: int, image_width: int, bbox_format: Literal["yolo", "coco", "pascal_voc"]):
  aug = A.Compose([
    A.OneOf([
      A.RandomResizedCrop(size=(image_height, image_width), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
      A.Resize(height=image_height, width=image_width)
    ], p=1.0),
    A.OneOf([
      A.HorizontalFlip(p=0.5),
      A.VerticalFlip(p=0.1),
      A.RandomRotate90(p=0.5),
      A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5, border_mode=0)
    ], p=0.8),
    A.OneOf([
      A.RandomBrightnessContrast(p=0.5),
      A.HueSaturationValue(p=0.5),
      A.RGBShift(p=0.5),
      A.ColorJitter(p=0.5)
    ], p=0.7),
    A.OneOf([
      A.MotionBlur(p=0.2),
      A.MedianBlur(blur_limit=3, p=0.1),
      A.GaussianBlur(blur_limit=3, p=0.1),
      A.GaussNoise(p=0.2),
    ], p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
  ],
  bbox_params=A.BboxParams(format=bbox_format,
                          label_fields=["class_labels"],
                          filter_invalid_bboxes=True))

  return aug

def get_mosaic_transform(grid_yx=(2, 2), target_size=(640, 480)):
  transform = A.Compose(
    [
      A.Mosaic(
        grid_yx=grid_yx,
        cell_shape=(512, 512),
        fit_mode="contain",
        target_size=target_size,
        metadata_key="mosaic_metadata",
        p=1,
      ),
    ],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["bbox_labels"]),
    p=1,
  )
  return transform