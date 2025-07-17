import torch
# for now just import the model
# from torchvision.models.detection.keypoint_rcnn import keypointrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2

from dataset.VisDroneDS.VisDroneDS import VisDroneDS
from torch.utils.data import DataLoader
from engine.trainer import BaseTrainer

device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2

if __name__ == "__main__":
  train_ds = VisDroneDS(root="data/TEST_VisDrone/train")
  train_dl = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, collate_fn=train_ds.collate_fn)

  val_ds = VisDroneDS(root="data/TEST_VisDrone/val")
  val_dl = DataLoader(dataset=val_ds, batch_size=BATCH_SIZE, collate_fn=val_ds.collate_fn)

  # x = [torch.rand((3, 1080, 1920)).to(device) for _ in range(1)]
  # out = model(x)
  # print(out)

  # model = keypointrcnn_resnet50_fpn(num_classes=10).to(device)
  model = fasterrcnn_resnet50_fpn_v2(num_classes=11)
  trainer = BaseTrainer(model=model, device=device, train_dataloader=train_dl, val_dataloader=val_dl, epochs=10)
  trainer.train()