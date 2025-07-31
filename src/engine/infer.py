import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from engine.utils import run_inference
import torch

def infer_and_display_image(image_path, model, device, image_size=640, conf_thresh=-0.1):
  raw_image = cv2.imread(image_path)
  image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

  transform = A.Compose([
    A.LongestMaxSize(max_size=image_size),
    A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
  ])

  transformed = transform(image=image_rgb)
  tensor = transformed["image"].unsqueeze(0).to(torch.float32).to(device)

  results = run_inference(
    model=model,
    device=device,
    inputs=tensor,
    image_size=image_size,
    empty_class_id=0,
    scale_boxes=True
  )

  boxes, scores, class_ids = results[0]

  print(boxes)

  vis_image = cv2.resize(raw_image, (image_size, image_size))
  # vis_image = image
  for box, score, class_id in zip(boxes, scores, class_ids):
    if score < conf_thresh:
      continue
    x1, y1, x2, y2 = map(int, box.tolist())
    label = f"Person: {score:.2f}"
    cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(vis_image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

  cv2.imshow("Prediction", vis_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()