import os
import torch
from typing import Dict

VisDrone_CLASS_NAMES = {
  1: 'pedestrian',
  2: 'people',
  3: 'bicycle',
  4: 'car',
  5: 'van',
  6: 'truck',
  7: 'tricycle',
  8: 'ajning-tricycle', # no clue what ajning means but yeah
  9: 'bus',
  10: 'motor'
}

def parse_VisDrone_annotations_file(annotations_file_path:str) -> Dict[torch.tensor, torch.tensor]:
  '''
  intput: 
    - a path to the annotations.txt file with VisDrone style annotations
  returns:
    - targets dict with "boxes" and "labels"; both a torch.tensor and boxes are XYXY
  '''
  assert os.path.exists(annotations_file_path)

  boxes, labels = [], []

  # open .txt annotations file to read
  with open(annotations_file_path, 'r') as f:
    for line in f:
      parts = line.strip().split(',')

      xmin = float(parts[0])
      ymin = float(parts[1])
      width = float(parts[2])
      height = float(parts[3])

      class_id = int(parts[5])
      
      # class_id == 0 means it is background, 11 is I have no idea what, so we skip these two
      if class_id == 0 or class_id == 11:
        continue

      # class_id += 1

      xmax = xmin + width
      ymax = ymin + height
      box = [xmin, ymin, xmax, ymax]
      # box = [xmin, ymin, height, width]

      boxes.append(box)
      labels.append(class_id)

  # close the .txt file
  f.close()

  boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
  labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64)

  return {"boxes": boxes, "labels": labels}

def collate_fn_simple(batch):
  ret = {}
  images, targets, img_paths = [], [], []

  for image, target, img_path in batch:
    images.append(image)
    targets.append(target)
    img_paths.append(img_path)

  ret = {"images": images, "targets": targets, "img_paths": img_paths}
  return ret

def collate_fn_tensor_stack(batch):
  images = []
  targets = []
  batch_len = []
  img_paths = []

  for image, target, img_path in batch:
    img_paths.append(img_path)
    n = target["boxes"].shape[0]
    if n > 0:
      labels = target["labels"].unsqueeze(1).float()
      boxes = target["boxes"]
      labeled_boxes = torch.cat([labels, boxes], dim=1)
      targets.append(labeled_boxes)
      batch_len.append(n)
    else:
      batch_len.append(0)

    images.append(image)

  images = torch.stack(images)
  if targets:
    target_tensor = torch.cat(targets, dim=0)
  else:
    target_tensor = torch.zeros((0, 5), dtype=torch.float32)

  return images, target_tensor, batch_len, img_paths