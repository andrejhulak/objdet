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
      if s > 0.1:
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

@torch.no_grad()
def test_video(model, postprocessors, video_path, device, output_path=None):
  model.eval()
  transform = A.Compose([
    A.Resize(height=480, width=640),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.ToTensorV2()
  ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], filter_invalid_bboxes=True))
  
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    return
  
  fps = int(cap.get(cv2.CAP_PROP_FPS))
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  
  print(f"Processing video: {video_path}")
  print(f"FPS: {fps}, Resolution: {width}x{height}, Total frames: {total_frames}")
  
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = None
  if output_path:
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
  
  frame_count = 0
  mean = torch.tensor([0.485, 0.456, 0.406], device="cpu").view(3, 1, 1)
  std = torch.tensor([0.229, 0.224, 0.225], device="cpu").view(3, 1, 1)
  
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    
    frame_count += 1
    print(f"Processing frame {frame_count}/{total_frames}", end='\r')
    
    original_frame = frame.copy()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    sample = {'image': img, 'bboxes': [], 'class_labels': []}
    sample = transform(**sample)
    input = sample['image'].unsqueeze(0).to(device)
    outputs = model(input)
    img_size = torch.tensor(input.shape[-2:]).unsqueeze(0).to(device)
    results = postprocessors['bbox'](outputs, img_size, not_to_xyxy=False)
    
    final_res = []
    for i, result in enumerate(results):
      scores = result['scores'].tolist()
      labels = result['labels'].tolist()
      boxes = result['boxes'].tolist()
      for s, l, b in zip(scores, labels, boxes):
        if s > 0.2:
          itemdict = {
            "category_id": l,
            "bbox": b,
            "score": s,
          }
          final_res.append(itemdict)
    
    top_preds = sorted(final_res, key=lambda x: x['score'], reverse=True)[:10]
    
    scale_x = width / 640
    scale_y = height / 480
    
    for pred in top_preds:
      x0, y0, x1, y1 = pred['bbox']
      x0 = int(x0 * scale_x)
      y0 = int(y0 * scale_y)
      x1 = int(x1 * scale_x)
      y1 = int(y1 * scale_y)
      
      cv2.rectangle(original_frame, (x0, y0), (x1, y1), color=(0, 255, 0), thickness=2)
      label_text = f"{classes[pred['category_id']]}:{pred['score']:.2f}"
      cv2.putText(original_frame, label_text, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    if output_path and out:
      out.write(original_frame)
    
    cv2.imshow("Video Detection", original_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  
  cap.release()
  if out:
    out.release()
  cv2.destroyAllWindows()
  print(f"\nProcessed {frame_count} frames")
  if output_path:
    print(f"Output saved to: {output_path}")