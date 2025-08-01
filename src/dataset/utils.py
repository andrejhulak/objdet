import numpy as np

def calculate_iou(box1, box2):
  x1_min = box1[0] - box1[2] / 2
  y1_min = box1[1] - box1[3] / 2
  x1_max = box1[0] + box1[2] / 2
  y1_max = box1[1] + box1[3] / 2
  
  x2_min = box2[0] - box2[2] / 2
  y2_min = box2[1] - box2[3] / 2
  x2_max = box2[0] + box2[2] / 2
  y2_max = box2[1] + box2[3] / 2
  
  inter_x_min = max(x1_min, x2_min)
  inter_y_min = max(y1_min, y2_min)
  inter_x_max = min(x1_max, x2_max)
  inter_y_max = min(y1_max, y2_max)
  
  if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
    return 0.0
  
  inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
  box1_area = box1[2] * box1[3]
  box2_area = box2[2] * box2[3]
  union_area = box1_area + box2_area - inter_area
  
  return inter_area / union_area if union_area > 0 else 0.0

def remove_duplicate_bboxes(bboxes, class_labels, iou_threshold):
  if len(bboxes) <= 1:
    return bboxes, class_labels
  
  bboxes = np.array(bboxes)
  class_labels = np.array(class_labels)
  keep_indices = []
  
  for i in range(len(bboxes)):
    should_keep = True
    for j in keep_indices:
      if class_labels[i] == class_labels[j]:
        iou = calculate_iou(bboxes[i], bboxes[j])
        if iou > iou_threshold:
          should_keep = False
          break
    if should_keep:
      keep_indices.append(i)
  
  return bboxes[keep_indices].tolist(), class_labels[keep_indices].tolist()