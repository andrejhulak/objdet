import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

num_classes = 10

class ArmaDS(Dataset):
  def __init__(self, root, image_size=640, max_boxes=100, augment=True):
    super().__init__()
    self.root = root
    self.image_dir = os.path.join(root, 'images')
    self.label_dir = os.path.join(root, 'labels')
    self.image_size = image_size
    self.max_boxes = max_boxes
    self.augment = augment
    self.image_paths = sorted([
      os.path.join(self.image_dir, fname)
      for fname in os.listdir(self.image_dir)
      if fname.endswith(('.jpg', '.png'))
    ])

    # self.strong_aug = A.Compose([
    #   A.OneOf([
    #     A.GridDropout(ratio=0.5, p=1.0)
    #   ], p=1.0),
    #   A.OneOf([
    #     A.ColorJitter(p=1.0),
    #     A.RandomBrightnessContrast(p=1.0),
    #     A.HueSaturationValue(p=1.0)
    #   ], p=1.0),
    #   A.HorizontalFlip(p=0.5),
    #   A.Resize(image_size, image_size),
    #   ToTensorV2()
    # ], bbox_params=A.BboxParams(
    #   format='yolo',
    #   label_fields=['class_labels'],
    #   filter_invalid_bboxes=True
    # ))

    self.strong_aug = A.Compose([A.Resize(image_size, image_size),
                                 A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                 A.ToTensorV2()])

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
      # class_labels = [cls_lbl for cls_lbl in class_labels]
    else:
      bboxes = []
      class_labels = []

    if self.augment:
      transformed = self.strong_aug(image=image, bboxes=bboxes, class_labels=class_labels)
      image = transformed['image']
      bboxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
      class_labels = torch.tensor(transformed['class_labels'], dtype=torch.int64)

    num_boxes = bboxes.shape[0]
    padded_boxes = torch.zeros((self.max_boxes, 4), dtype=torch.float32)
    padded_classes = torch.full((self.max_boxes,), num_classes + 1, dtype=torch.int64)
    mask = torch.zeros((self.max_boxes,), dtype=torch.uint8)

    if num_boxes > 0:
      count = min(self.max_boxes, num_boxes)
      padded_boxes[:count] = bboxes[:count]
      padded_classes[:count] = class_labels[:count]
      mask[:count] = 1

    return image.to(torch.float32), (padded_classes, padded_boxes, mask, torch.tensor([idx], dtype=torch.int64))

def collate_fn(inputs):
  input_ = torch.stack([i[0] for i in inputs])
  classes = torch.stack([i[1][0] for i in inputs])
  boxes = torch.stack([i[1][1] for i in inputs])
  masks = torch.stack([i[1][2].to(dtype=torch.long) for i in inputs])
  image_ids = torch.stack([i[1][3] for i in inputs])
  return input_, (classes, boxes, masks, image_ids)