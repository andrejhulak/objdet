import torch
from dataset.base_ds import ArmaDS, collate_fn
from torch.utils.data.dataloader import DataLoader
from models.dino.dino import build_dino
from engine import test_single_image
from tqdm import tqdm

import DINO_4scale as args

device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
n_epochs = 40

if __name__ == "__main__":
  model, criterion, postprocessors = build_dino(args)
  # model.load_state_dict(torch.load("pth/dino_arma_model_1.pth"))
  model = model.to(device).train()
  criterion.train()

  train_ds = ArmaDS(root="data/arma")
  train_dl = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

  print("Starting training...")
  for epoch in range(n_epochs):
    total_loss = 0
    for input, targets in tqdm(train_dl):
      input = input.to(device)
      targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
      optimizer.zero_grad()

      outputs = model(input, targets)
      loss_dict = criterion(outputs, targets)
      weight_dict = criterion.weight_dict
      losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
      total_loss += losses.item()
      
      losses.backward()
      grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
      optimizer.step()

    total_loss /= len(train_ds)
    print(f"Epoch {epoch}: Total Loss = {total_loss:.4f}")

  torch.save(model.state_dict(), "pth/dino_swinL_mod.pth")

  # test_single_image(model, postprocessors, "data/arma/images/frame_0.jpg", device)
  test_single_image(model, postprocessors, "data/drone_pic.jpg", device)