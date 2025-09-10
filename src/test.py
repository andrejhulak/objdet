import torch
from dataset.base_ds import ArmaDS, collate_fn
from torch.utils.data.dataloader import DataLoader
from models.dino.dino import build_dino
from engine import test_single_image, test_video

import DINO_4scale as args

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
  model, criterion, postprocessors = build_dino(args)
  model = model.to(device).train()
  model.load_state_dict(torch.load("pth/ddinov3gpt.pth"))

  # test_single_image(model, postprocessors, "data/og ds/images/frame_0.jpg", device)
  # test_single_image(model, postprocessors, "data/drone_pic_3.jpg", device)
  test_video(model, postprocessors, "data/vid_final.mp4", device)