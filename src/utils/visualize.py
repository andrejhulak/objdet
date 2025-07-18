import torch
import os
import cv2
from typing import Dict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 0, 255)

def visualize_gt_batch(dataloader:torch.utils.data.DataLoader, 
                       class_names:Dict[int, str]) -> None:
  '''
  input:
    - dataloader: dataloder to visualize images from
    - class names: names of the classes in the dataset and their corresponding int
  '''

  for _, targets, image_paths in dataloader:
    assert isinstance(dataloader.batch_size, int)
    batch_size = dataloader.batch_size

    for i in range(batch_size):
      image = cv2.imread(image_paths[i])
      boxes = targets[i]["boxes"]
      labels = targets[i]["labels"]

      for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box)
        # class_name = class_names[int(label)]
        # label_text = f"{class_name}"

        cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, 2)
        # cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 2)

      cv2.imshow(f"Prediction {os.path.basename(image_paths[i])}", image)
      cv2.waitKey(5000)

  cv2.destroyAllWindows()


def visualize_predictions(model, dataloader, class_names, confidence_threshold=0.5):
    model.eval()
    with torch.no_grad():
      for batch in dataloader:
        images = batch["images"]
        image_paths = batch["img_paths"]
        images = [img.to(device) for img in images]
        outputs = model(images)

        for i, output in enumerate(outputs):
          if output is not None and len(output) > 0:
            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy().astype(int)

            image = cv2.imread(image_paths[i])
            h, w = image.shape[:2]

            for box, score, label in zip(boxes, scores, labels):
              if score > confidence_threshold:
                x1, y1, x2, y2 = map(int, box)
                class_name = class_names[label]
                confidence_text = f"{score:.2f}"
                label_text = f"{class_name} {confidence_text}"

                cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, 2)
                cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 2)

            cv2.imshow(f"Prediction {os.path.basename(image_paths[i])}", image)
            cv2.waitKey(5000)
        cv2.destroyAllWindows()