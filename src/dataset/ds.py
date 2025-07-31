import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ArmaDS(Dataset):
  def __init__(self, root, image_size=(640, 480), augment=True):
    super().__init__()
    self.root = root
    self.image_dir = os.path.join(root, 'images')
    self.label_dir = os.path.join(root, 'labels')
    self.image_size = image_size
    self.augment = augment
    self.image_paths = sorted([
      os.path.join(self.image_dir, fname)
      for fname in os.listdir(self.image_dir)
      if fname.endswith(('.jpg', '.png'))
    ])
    self.h, self.w = image_size
    self.aug = A.Compose([
      A.OneOf([
        A.RandomResizedCrop(size=(self.h, self.w), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        A.Resize(height=self.h, width=self.w)
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
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
      ], p=0.3),

      A.Normalize(mean=(0.485, 0.456, 0.406),
                  std=(0.229, 0.224, 0.225)),

      ToTensorV2()
    ],
    bbox_params=A.BboxParams(format="yolo",
                            label_fields=["class_labels"],
                            filter_invalid_bboxes=True))

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, idx):
    img_path = self.image_paths[idx]
    label_path = os.path.join(self.label_dir, os.path.splitext(os.path.basename(img_path))[0] + '.txt')
    
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if os.path.exists(label_path):
      with open(label_path, 'r') as f:
        lines = f.readlines()
      gt = [list(map(float, line.strip().split())) for line in lines]
    else:
      gt = []
    
    if len(gt) > 0:
      labels = np.array(gt)
      if not ((labels[:, 1:] >= 0).all() and (labels[:, 1:] <= 1).all()):
        raise ValueError(f'Invalid bbox values in {label_path}')
      bboxes = labels[:, 1:]
      class_labels = labels[:, 0].astype(np.int32).tolist()
      # class_labels = [cls_label + 1 for cls_label in class_labels]
    else:
      bboxes = []
      class_labels = []
    
    if self.augment:
      transformed = self.aug(image=image, bboxes=bboxes, class_labels=class_labels)
      image = transformed['image']
      bboxes = transformed['bboxes']
      class_labels = transformed['class_labels']
    
    target = {
      'labels': torch.tensor(class_labels, dtype=torch.long),
      'boxes': torch.tensor(bboxes, dtype=torch.float32) if len(bboxes) > 0 else torch.zeros((0, 4), dtype=torch.float32),
      'image_id': torch.tensor(idx, dtype=torch.long),
      'orig_size': torch.tensor([self.w, self.h], dtype=torch.long),
      'size': torch.tensor([self.w, self.h], dtype=torch.long)
    }

    return image.to(torch.float32), target

def collate_fn(batch):
  images = torch.stack([item[0] for item in batch])
  targets = [item[1] for item in batch]
  return images, targets