import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

from dataset.transforms import get_transforms
from dataset.utils import remove_duplicate_bboxes

class ArmaDS(Dataset):
  def __init__(self, root, image_size=(640, 480), augment=True):
    super().__init__()
    self.root = root
    self.image_dir = os.path.join(root, 'images')
    self.label_dir = os.path.join(root, 'labels')

    assert os.path.exists(self.image_dir)
    assert os.path.exists(self.label_dir)

    self.image_size = image_size
    self.w, self.h = image_size

    self.augment = augment
    self.duplicate_boxes_iou_threshold = 0.7

    self.image_paths = sorted([
      os.path.join(self.image_dir, fname)
      for fname in os.listdir(self.image_dir)
      if fname.endswith(('.jpg', '.png'))
    ])

    self.aug = get_transforms(image_height=self.h,
                              image_width=self.w,
                              bbox_format="yolo")

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
      bboxes = labels[:, 1:].tolist()
      class_labels = labels[:, 0].astype(np.int32).tolist()
      bboxes, class_labels = remove_duplicate_bboxes(bboxes=bboxes,
                                                     class_labels=class_labels,
                                                     iou_threshold=self.duplicate_boxes_iou_threshold)
    else:
      bboxes = []
      class_labels = []
    
    if self.augment:
      transformed = self.aug(image=image, bboxes=bboxes, class_labels=class_labels)
      image = transformed['image'].to(torch.float32)
      bboxes = transformed['bboxes']
      class_labels = transformed['class_labels']
      if len(bboxes) > 1:
        bboxes, class_labels = remove_duplicate_bboxes(bboxes=bboxes,
                                                       class_labels=class_labels,
                                                       iou_threshold=self.duplicate_boxes_iou_threshold)
    
    target = {
      'labels': torch.tensor(class_labels, dtype=torch.long),
      'boxes': torch.tensor(bboxes, dtype=torch.float32) if len(bboxes) > 0 else torch.zeros((0, 4), dtype=torch.float32),
      'image_id': torch.tensor(idx, dtype=torch.long),
      'orig_size': torch.tensor([self.w, self.h], dtype=torch.long),
      'size': torch.tensor([self.w, self.h], dtype=torch.long)
    }

    return image, target

def collate_fn(batch):
  images = torch.stack([item[0] for item in batch])
  targets = [item[1] for item in batch]
  return images, targets