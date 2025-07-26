import torch
import random
# for now just import the model
# from torchvision.models.detection.keypoint_rcnn import keypointrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2
from models.sparse_rcnn.sparse_rcnn import SparseRCNN
from utils.visualize import visualize_gt_batch

from dataset.VisDroneDS.VisDroneDS import VisDroneDS
from dataset.ArmaDS.ArmaDS import ArmaDS
from torch.utils.data import DataLoader
from engine.trainer import BaseTrainer

device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1

class_names = {
  0 : "person",
  1 : "vehicle"
}

if __name__ == "__main__":
  train_ds = ArmaDS(root="data/TEST_Arma/train")
  train_dl = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, collate_fn=train_ds.collate_fn)

  # val_ds = VisDroneDS(root="data/TEST_VisDrone/train")
  # val_dl = DataLoader(dataset=val_ds, batch_size=BATCH_SIZE, collate_fn=val_ds.collate_fn)

  # for batch in train_dl:
  #   print(batch)
  visualize_gt_batch(train_dl, class_names)