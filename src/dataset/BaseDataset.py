import torch
import os

class BaseDataset(torch.utils.data.Dataset):
  def __init__(self, root:str):
    super().__init__()
    self.root = root
    self.class_names = None
    self.collate_fn = None

    self.images_dir = os.path.join(root, "images")
    self.annotations_dir = os.path.join(root, "annotations")
    self.imgs = sorted([
      f for f in os.listdir(self.images_dir)
      if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

  def __getitem__(self, idx):
    pass

  def __len__(self):
    pass