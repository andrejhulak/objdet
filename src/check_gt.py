import torch
import cv2
import numpy as np
from dataset.base_ds import ArmaDS
from torchvision.transforms.functional import to_pil_image
from dataset.mosaic_ds import MosaicDataset

def draw_yolo_boxes(image_tensor, boxes, frame_num, labels=None):
  mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
  std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
  
  image_denorm = image_tensor * std + mean
  image_denorm = torch.clamp(image_denorm, 0, 1)
  
  image = image_denorm.permute(1, 2, 0).numpy()
  image = (image * 255).astype(np.uint8)
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  
  h, w, _ = image.shape
  print(f"Image shape: {image.shape}, Tensor shape: {image_tensor.shape}")
  
  for i, box in enumerate(boxes):
    cx, cy, bw, bh = box.tolist()
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    if labels is not None:
      label = int(labels[i].item())
      cv2.putText(image, f"Class: {label}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
  
  cv2.imshow(f"Frame {frame_num}", image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

if __name__ == "__main__":
  ds = ArmaDS(root="data/arma")
  ds_mosaic = MosaicDataset(dataset=ds)

  for i in range(len(ds_mosaic)):
    img, target = ds_mosaic[i]
    # img, target = ds[i]
    boxes = target["boxes"]
    labels = target.get("labels", None)
    draw_yolo_boxes(img, boxes, i, labels)