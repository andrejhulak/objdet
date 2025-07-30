import torch
import torchvision.transforms as T
from torchvision import datasets
from torch.nn.utils.rnn import pad_sequence
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
import torch
from torchvision.ops import box_convert

def pad_bboxes_with_mask(bboxes_list, max_boxes, device, pad_value=0.0):
  """
  Pads bounding boxes to a fixed size and returns a mask.

  Args:
    bboxes_list (list of torch.Tensor): List of (num_boxes, 4) tensors.
    max_boxes (int): Fixed number of boxes to pad/truncate to.
    pad_value (float, optional): Padding value. Defaults to 0.0.

  Returns:
    tuple: (padded_boxes, mask)
      - padded_boxes (torch.Tensor): (batch_size, max_boxes, 4)
      - mask (torch.Tensor): (batch_size, max_boxes), 1 for real boxes, 0 for padded ones.
  """
  padded_boxes = pad_sequence(bboxes_list, batch_first=True, padding_value=pad_value)

  if padded_boxes.shape[1] > max_boxes:
    padded_boxes = padded_boxes[:, :max_boxes, :]
  elif padded_boxes.shape[1] < max_boxes:
    extra_pad = torch.full(
      (padded_boxes.shape[0], max_boxes - padded_boxes.shape[1], 4),
      pad_value,
      device=device,
    )
    padded_boxes = torch.cat([padded_boxes, extra_pad], dim=1)

  mask = (padded_boxes[:, :, 0] != pad_value).float()

  return padded_boxes, mask

def pad_classes(class_list, max_classes, device, pad_value=0):
  """
  Pads class labels to a fixed size.

  Args:
    class_list (list of torch.Tensor): List of (num_classes,) tensors.
    max_classes (int): Fixed number of classes to pad/truncate to.
    pad_value (int, optional): Padding value for classes.

  Returns:
    tuple: (padded_classes, mask)
      - padded_classes (torch.Tensor): (batch_size, max_classes)
  """
  padded_classes = pad_sequence(class_list, batch_first=True, padding_value=pad_value)

  if padded_classes.shape[1] > max_classes:
    padded_classes = padded_classes[:, :max_classes]
  elif padded_classes.shape[1] < max_classes:
    extra_pad = torch.full(
      (padded_classes.shape[0], max_classes - padded_classes.shape[1]),
      pad_value,
      device=device,
    )
    padded_classes = torch.cat([padded_classes, extra_pad], dim=1)

  return padded_classes