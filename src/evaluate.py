import torch
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

@torch.no_grad()
def evaluate_model_map(model, postprocessors, dataloader, device, score_threshold=0.2, max_detections=100):
  model.eval()
  
  map_metric = MeanAveragePrecision(
    box_format="xyxy",
    iou_type="bbox",
    iou_thresholds=None,
    rec_thresholds=None,
    max_detection_thresholds=[1, 10, 100],
    class_metrics=True,
  )
  
  print(f"Starting evaluation on {len(dataloader)} batches...")
  
  for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="Evaluating")):
    images = images.to(device)
    targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
    
    outputs = model(images)
    
    img_sizes = torch.tensor([img.shape[-2:] for img in images]).to(device)
    
    results = postprocessors['bbox'](outputs, img_sizes, not_to_xyxy=False)
    
    preds = []
    gts = []
    
    for i, (result, target) in enumerate(zip(results, targets)):
      valid_indices = result['scores'] > score_threshold
      
      if valid_indices.sum() == 0:
        pred_dict = {
          'boxes': torch.empty((0, 4), device=device),
          'scores': torch.empty((0,), device=device),
          'labels': torch.empty((0,), device=device, dtype=torch.int64),
        }
      else:
        sorted_indices = torch.argsort(result['scores'][valid_indices])
        top_indices = sorted_indices[:max_detections]
        
        valid_scores = result['scores'][valid_indices][top_indices]
        valid_labels = result['labels'][valid_indices][top_indices]
        valid_boxes = result['boxes'][valid_indices][top_indices]
        
        pred_dict = {
          'boxes': valid_boxes,
          'scores': valid_scores,
          'labels': valid_labels,
        }
      
      preds.append(pred_dict)

      gt_boxes_yolo = target['boxes']

      img_w = img_sizes[0][1]
      img_h = img_sizes[0][0]

      if gt_boxes_yolo.shape[0] > 0:
        x_center = gt_boxes_yolo[:, 0] * img_w
        y_center = gt_boxes_yolo[:, 1] * img_h
        width = gt_boxes_yolo[:, 2] * img_w
        height = gt_boxes_yolo[:, 3] * img_h
        
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        gt_boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
      
      gt_dict = {
        'boxes': gt_boxes_xyxy,
        'labels': target['labels'],
      }
      gts.append(gt_dict)
    
    map_metric.update(preds, gts)
  
  map_results = map_metric.compute()
  print(map_results)
  
  print("\n" + "="*50)
  print("EVALUATION RESULTS")
  print("="*50)
  print(f"mAP (IoU=0.50:0.95): {map_results['map']:.4f}")
  print(f"mAP (IoU=0.50):     {map_results['map_50']:.4f}")
  print(f"mAP (IoU=0.75):     {map_results['map_75']:.4f}")
  print(f"mAP (small):        {map_results['map_small']:.4f}")
  print(f"mAP (medium):       {map_results['map_medium']:.4f}")
  print(f"mAP (large):        {map_results['map_large']:.4f}")

  return map_results

if __name__ == "__main__":
  from dataset.base_ds import ArmaDS, collate_fn
  from torch.utils.data.dataloader import DataLoader
  from models.dino.dino import build_dino
  import DINO_4scale as args
  
  device = "cuda" if torch.cuda.is_available() else "cpu"
  BATCH_SIZE = 1
  
  model, criterion, postprocessors = build_dino(args)
  model.load_state_dict(torch.load("pth/ddinov3.pth"))
  # model.load_state_dict(torch.load("pth/ddinov3_50temp.pth"))
  model = model.to(device)
  
  test_ds = ArmaDS(root="data/test", augment=False)
  test_dl = DataLoader(dataset=test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
  
  map_results = evaluate_model_map(
      model=model,
      postprocessors=postprocessors,
      dataloader=test_dl,
      device=device,
      score_threshold=0.1,
      max_detections=100
  )