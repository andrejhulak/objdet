import torch
import random
# for now just import the model
# from torchvision.models.detection.keypoint_rcnn import keypointrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2
from models.sparse_rcnn.sparse_rcnn import SparseRCNN

from dataset.VisDroneDS.VisDroneDS import VisDroneDS
from torch.utils.data import DataLoader
from engine.trainer import BaseTrainer

device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1

if __name__ == "__main__":
  train_ds = VisDroneDS(root="data/TEST_VisDrone/train")
  train_dl = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, collate_fn=train_ds.collate_fn)

  val_ds = VisDroneDS(root="data/TEST_VisDrone/train")
  val_dl = DataLoader(dataset=val_ds, batch_size=BATCH_SIZE, collate_fn=val_ds.collate_fn)

  model = SparseRCNN()
  model.to(device)

  # b_size = 2
  # x = torch.rand((b_size, 3, 640, 640)).to(device)
  # targets = []
  # batch_len = []

  # for _ in range(b_size):
  #   n_objects = random.randint(5, 10)
  #   boxes = torch.rand((n_objects, 4)) * 640
  #   boxes[:, 2:] = torch.maximum(boxes[:, :2] + 1, boxes[:, 2:])
  #   labels = torch.randint(0, 10, (n_objects, 1)).float()
  #   labeled_boxes = torch.cat([labels, boxes], dim=1)
  #   targets.append(labeled_boxes)
  #   batch_len.append(n_objects)

  # target_tensor = torch.cat(targets, dim=0).to(device)
  # batch_len_tensor = torch.tensor(batch_len).to(device)

  # out = model(x, targets={"target": target_tensor, "batch_len": batch_len_tensor})


  # model = keypointrcnn_resnet50_fpn(num_classes=10).to(device)
  # model = fasterrcnn_resnet50_fpn_v2(num_classes=11)
  trainer = BaseTrainer(model=model, device=device, train_dataloader=train_dl, val_dataloader=val_dl, epochs=2000)
  trainer.train()
  trainer.evaluate()