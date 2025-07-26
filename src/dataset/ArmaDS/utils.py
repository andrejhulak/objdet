import os
import torch
from typing import Dict

def parse_arma_DS_annotations(annotations_filepath: str, img_width: int = 1920, img_height: int = 1080) -> Dict[str, torch.Tensor]:
  '''
  input: 
    - annotations_filepath: path to YOLO-style annotation file (cx, cy, w, h — all normalized)
  returns:
    - targets dict with "boxes" and "labels"; both as torch.Tensor
      boxes are in absolute pixel coordinates (x_min, y_min, x_max, y_max)
  '''
  assert os.path.exists(annotations_filepath)

  boxes, labels = [], []

  with open(annotations_filepath, 'r') as f:
    for line in f:
      parts = line.strip().split()
      if len(parts) != 5:
        continue

      class_id = int(parts[0])
      cx, cy, w, h = map(float, parts[1:])

      x1 = (cx - w / 2) * img_width
      y1 = (cy - h / 2) * img_height
      x2 = (cx + w / 2) * img_width
      y2 = (cy + h / 2) * img_height

      boxes.append([x1, y1, x2, y2])
      labels.append(class_id)

  boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
  labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64)

  return {"boxes": boxes, "labels": labels}
