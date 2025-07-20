import cv2
import os
import torch

BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)

def visualize_predictions(model, dataloader, class_names, device, confidence_threshold=0.75):
  model.eval()
  with torch.no_grad():
    for batch in dataloader:
      images, targets, batch_len, img_paths = batch
      images = images.to(device)

      outputs = model(images)

      for i, output in enumerate(outputs):
        output = outputs[output]
        boxes = output[i][:, :4].cpu()
        scores = output[i][:, 4].detach().cpu()
        labels = output[i][:, 5].detach().cpu()
        if boxes.numel() == 0:
          continue
        image = cv2.imread(img_paths[i])
        for box, score, label in zip(boxes, scores, labels):
          if score > confidence_threshold:
            x1, y1, x2, y2 = map(int, box)
            label_text = f"{class_names[int(label)]} {score:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, 2)
            cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 2)
        cv2.imshow(f"{os.path.basename(img_paths[i])}", image)
        cv2.waitKey(5000)
    cv2.destroyAllWindows()
