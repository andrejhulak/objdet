import os
import torch
from typing import Dict

import os
import torch
from typing import Dict

def parse_Arma_annotations_file(annotations_file_path: str, image_width: int = 1920, image_height: int = 1080) -> Dict[str, torch.Tensor]:
  assert os.path.exists(annotations_file_path)

  keypoints = []

  with open(annotations_file_path, 'r') as f:
    for line in f:
      parts = line.strip().split()
      if len(parts) != 2:
        continue  # skip malformed lines

      norm_x, norm_y = float(parts[0]), float(parts[1])
      x = norm_x * image_width
      y = norm_y * image_height

      keypoints.append([x, y])

  keypoints = torch.tensor(keypoints, dtype=torch.float32) if keypoints else torch.zeros((0, 2), dtype=torch.float32)

  return {"keypoints": keypoints}


def collate_fn_simple(batch):
  ret = {}
  images, targets, img_paths = [], [], []

  for image, target, img_path in batch:
    images.append(image)
    targets.append(target)
    img_paths.append(img_path)

  ret = {"images": images, "targets": targets, "img_paths": img_paths}
  return ret