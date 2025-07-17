import torch
import os
import cv2
from typing import Dict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
POINT_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 0, 255)

def visualize_batch(dataloader: torch.utils.data.DataLoader, class_names: Dict[int, str] = None) -> None:
  for _, batch_targets, image_paths in dataloader:
    batch_size = dataloader.batch_size

    for i in range(batch_size):
      image = cv2.imread(image_paths[i])
      # image = cv2.resize(image, (1333, 800))
      target = batch_targets["keypoints"][0]
      print(target)

      # keypoints = target.get("keypoints", torch.empty((0, 2)))

      for point in target:
        x, y = map(int, point)
        cv2.circle(image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)

      cv2.imshow(f"Keypoints: {os.path.basename(image_paths[i])}", image)
      cv2.waitKey(5000)
      cv2.destroyAllWindows()