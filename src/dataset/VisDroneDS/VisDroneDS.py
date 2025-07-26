import os
import torch
from torchvision.io import read_file, decode_image

from .utils import parse_VisDrone_annotations_file, VisDrone_CLASS_NAMES, collate_fn_simple, collate_fn_tensor_stack
from dataset.BaseDataset import BaseDataset

from torchvision.transforms import v2

class VisDroneDS(BaseDataset):
  def __init__(self, root:str):
    super().__init__(root)
    self.class_names = VisDrone_CLASS_NAMES
    self.collate_fn = collate_fn_tensor_stack

  def __getitem__(self, idx):
    img_file = self.imgs[idx]
    img_path = os.path.join(self.images_dir, img_file)
    # obtain the "name" of the image (#TODO think of a better name than "name")
    base_name = os.path.splitext(img_file)[0]

    img = decode_image(read_file(img_path)).to(torch.float32)
    resize = v2.Resize((1920, 1080))
    img = resize(img) / 255.0

    ann_path = os.path.join(self.annotations_dir, f"{base_name}.txt")
    targets = parse_VisDrone_annotations_file(annotations_file_path=ann_path)

    return img, targets, img_path

  def __len__(self):
    return len(self.imgs)