import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Literal, Tuple
from numpy import random

class Transforms():
  def __init__(self,
              image_size: Tuple[int, int],
              bbox_format: Literal["yolo", "pascal_voc", "coco"]
              ):
    assert image_size, bbox_format

    self.image_height = image_size[0]
    self.image_width = image_size[1]
    self.bbox_format = bbox_format

    self.basic_transforms = self.create_basic_transforms()

    self.mosaic = self.get_mosaic_transform(grid_yx=(2, 2))

  def create_basic_transforms(self):
    aug = A.Compose([
      A.OneOf([
        A.HorizontalFlip(p=0.5),
        # A.RandomResizedCrop(size=(self.image_height, self.image_width), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        A.VerticalFlip(p=0.1),
        A.RandomRotate90(p=0.5),
        # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5, border_mode=0)
      ], p=0.8),
      A.OneOf([
        A.GridDistortion()
      ], p=0.5),
      A.OneOf([
        A.PixelDropout(dropout_prob=random.rand() / 9.0)
      ], p=1),
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
      A.Resize(height=self.image_height, width=self.image_width),
      A.Normalize(mean=(0.485, 0.456, 0.406),
                  std=(0.229, 0.224, 0.225)),
      ToTensorV2()
    ],
    bbox_params=A.BboxParams(format=self.bbox_format,
                            label_fields=["class_labels"],
                            filter_invalid_bboxes=True))
    return aug

  def get_mosaic_transform(self, grid_yx: Tuple[int, int]):
    target_size = (self.image_height, self.image_width)
    mosaic = A.Compose(
      [
        A.Mosaic(
          grid_yx=grid_yx,
          cell_shape=(512, 512), # fine for now
          fit_mode="cover",
          target_size=target_size,
          metadata_key="mosaic_metadata",
          p=1,
        ),
        ToTensorV2()
      ],
      bbox_params=A.BboxParams(format="yolo",
                               label_fields=["class_labels"],
                               filter_invalid_bboxes=True),
      p=1,
    )
    return mosaic