import os
import torch
from torchvision.io import read_file, decode_image

from .utils.VisDroneDS_utils import parse_VisDrone_annotations_file, VisDrone_CLASS_NAMES, collate_fn_simple
from dataset.BaseDataset import BaseDataset

class VisDroneDS(BaseDataset):
  def __init__(self, root:str):
    super().__init__(root)
    self.class_names = VisDrone_CLASS_NAMES
    self.collate_fn = collate_fn_simple

  def __getitem__(self, idx):
    img_file = self.imgs[idx]
    img_path = os.path.join(self.images_dir, img_file)
    # obtain the "name" of the image (#TODO think of a better name than "name")
    base_name = os.path.splitext(img_file)[0]

    img = decode_image(read_file(img_path)).to(torch.float32)

    ann_path = os.path.join(self.annotations_dir, f"{base_name}.txt")
    targets = parse_VisDrone_annotations_file(annotations_file_path=ann_path)

    return img, targets, img_path

  def __len__(self):
    return len(self.imgs)