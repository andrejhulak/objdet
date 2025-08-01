import torch
from dataset.ds import ArmaDS, collate_fn
from torch.utils.data.dataloader import DataLoader
from models.dino.dino import build_dino
from engine import test_single_image

import DINO_4scale as args

device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1

if __name__ == "__main__":
  model, criterion, postprocessors = build_dino(args)
  model = model.to(device).train()
  # model.load_state_dict(torch.load("pth/dino_arma_model_1.pth"))
  model.load_state_dict(torch.load("pth/dino_swinL2.pth"))

  train_ds = ArmaDS(root="data/arma")
  train_dl = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

  # test_single_image(model, postprocessors, "data/arma/images/frame_1.jpg", device)
  test_single_image(model, postprocessors, "data/drone_pic_2.jpg", device)