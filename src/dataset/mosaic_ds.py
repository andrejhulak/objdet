import torch
from torch.utils.data.dataset import Dataset
from dataset.transforms import Transforms
import numpy as np

class MosaicDataset(Dataset):
  def __init__(self, dataset: Dataset):
    super().__init__()
    self.dataset = dataset
    self.h = self.dataset.h
    self.w = self.dataset.w

    self.transform = Transforms(image_size=(self.h, self.w),
                                bbox_format="yolo")

    self.mosaic = self.transform.mosaic

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    primary_example = self.dataset[idx]
    
    c, h, w = primary_example[0].shape

    primary_image = primary_example[0].reshape(h, w, c).numpy()
    primary_bboxes = primary_example[1]["boxes"].numpy()
    primary_labels = primary_example[1]["labels"].tolist()

    #TODO make this better
    other_example_list = []
    while len(other_example_list) <= 3:
      random_idx = np.random.randint(low=0, high=len(self.dataset))
      if idx != random_idx:
        other_example_image = self.dataset[random_idx][0].reshape(h, w, c).numpy() # get the image of the random example
        other_example_bboxes = self.dataset[random_idx][1]["boxes"].numpy() # get the bboxes of the random example
        other_example_labels = self.dataset[random_idx][1]["labels"].tolist() # get the labels of the random example
        other_example = {
          "image": other_example_image,
          "bboxes": other_example_bboxes,
          "class_labels": other_example_labels
        }
        other_example_list.append(other_example)

    transformed = self.mosaic(
      image=primary_image,
      bboxes=primary_bboxes,
      class_labels=primary_labels,
      mosaic_metadata=other_example_list
    )

    target = {
      "labels": torch.tensor(transformed ["class_labels"], dtype=torch.long),
      "boxes": torch.tensor(transformed["bboxes"], dtype=torch.float32) if len(transformed["bboxes"]) > 0 else torch.zeros((0, 4), dtype=torch.float32),
      "image_id": torch.tensor(idx, dtype=torch.long),
      "orig_size": torch.tensor([self.w, self.h], dtype=torch.long),
      "size": torch.tensor([self.w, self.h], dtype=torch.long)
    }

    image = transformed["image"].to(torch.float32)

    return image, target

def collate_fn(batch):
  images = torch.stack([x[0] for x in batch])
  targets = [x[1] for x in batch]
  return images, targets