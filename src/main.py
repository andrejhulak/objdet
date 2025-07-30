import torch
from dataset.ds import ArmaDS, collate_fn
from torch.utils.data.dataloader import DataLoader
from models.cond_detr import ConditionalDETR
from engine.train import train
from engine.infer import infer_and_display_image

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
BATCH_SIZE = 1

if __name__ == "__main__":
  train_ds = ArmaDS(root="data/arma")
  train_dl = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

  model = ConditionalDETR(
    d_model=224,
    n_classes=2,
    n_tokens=400, # for now
    n_layers=6,
    n_heads=2,
    n_queries=100,
    use_frozen_bn=False
  ).to(torch.float32).to(device)
  
  train(model=model,
        train_loader=train_dl,
        device=device,
        epochs=100,
        batch_size=BATCH_SIZE)

  infer_and_display_image(model=model, image_path="data/arma/images/frame_0.jpg", device=device, class_names=["person"])

  # x = torch.rand(2, 3, 640, 640).to(device)
  # out = model(x)
  # # the model returns a tuple of len() 2
  # # the first element is of (bsize, n_layers, n_queries, n_classes)
  # # the second element is of (bsize, n_layers, n_queries, 4)
  # gt_boxes = torch.rand(100, 4)