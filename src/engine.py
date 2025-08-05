import torch
import cv2
import numpy as np
import albumentations as A

classes = {
  0: "soldier"
}

@torch.no_grad()
def test_single_image(model, postprocessors, path, device):
  model.eval()

  transform = A.Compose([
    # A.Resize(height=800, width=1333),
    A.Resize(height=480, width=640),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.ToTensorV2()
  ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], filter_invalid_bboxes=True))

  img = cv2.imread(path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  h, w = img.shape[:2]
  sample = {'image': img, 'bboxes': [], 'class_labels': []}
  sample = transform(**sample)
  input = sample['image'].unsqueeze(0).to(device)

  outputs = model(input)
  img_size = torch.tensor(input.shape[-2:]).unsqueeze(0).to(device)
  results = postprocessors['bbox'](outputs, img_size, not_to_xyxy=False)

  final_res = []
  for i, result in enumerate(results):
    image_id = i
    scores = result['scores'].tolist()
    labels = result['labels'].tolist()
    boxes = result['boxes'].tolist()
    for s, l, b in zip(scores, labels, boxes):
      if s > 0.2:
        itemdict = {
          "image_id": int(image_id),
          "category_id": l,
          "bbox": b,
          "score": s,
        }
        final_res.append(itemdict)

  print(f"Found {len(final_res)} predictions")

  top_preds = sorted(final_res, key=lambda x: x['score'], reverse=True)[:10]
  for i, pred in enumerate(top_preds):
    print(f"Pred {i+1}: score={pred['score']:.3f}, class={pred['category_id']}, bbox={pred['bbox']}")

  mean = torch.tensor([0.485, 0.456, 0.406], device="cpu").view(3, 1, 1)
  std = torch.tensor([0.229, 0.224, 0.225], device="cpu").view(3, 1, 1)
  img = input[0].cpu() * std + mean
  img = img.permute(1, 2, 0).clamp(0, 1).numpy()
  img_cv = (img * 255).astype(np.uint8)
  img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

  for pred in top_preds:
    x0, y0, x1, y1 = map(int, pred['bbox'])
    cv2.rectangle(img_cv, (x0, y0), (x1, y1), color=(0, 255, 0), thickness=2)
    label_text = f"{classes[pred['category_id']]}:{pred['score']:.2f}"
    cv2.putText(img_cv, label_text, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

  cv2.imshow("Top Predictions", img_cv)
  cv2.waitKey(0)
  cv2.destroyAllWindows()