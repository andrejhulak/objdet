import torch
import torch.nn as nn
from models.losses.detr_loss import compute_sample_loss

def train(model, train_loader, device, epochs, batch_size):
  optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.00001, weight_decay=0.0001)
  model.train()
  print(f"Starting training for {epochs} epochs... Using device : {device}")

  scaler = torch.amp.GradScaler()

  losses = torch.tensor([], device=device)
  class_losses = torch.tensor([], device=device)
  box_losses = torch.tensor([], device=device)
  giou_losses = torch.tensor([], device=device)

  for epoch in range(epochs):
    for batch_idx, (input_, (tgt_cl, tgt_bbox, tgt_mask, _)) in enumerate(train_loader):
      input_ = input_.to(device)
      tgt_cl = tgt_cl.to(device)
      tgt_bbox = tgt_bbox.to(device)
      tgt_mask = tgt_mask.bool().to(device)

      with torch.amp.autocast(device_type=device):
        class_preds, bbox_preds = model(input_)

        loss = torch.tensor(0.0, device=device)
        loss_class_batch = torch.tensor(0.0, device=device)
        loss_bbox_batch = torch.tensor(0.0, device=device)
        loss_giou_batch = torch.tensor(0.0, device=device)

        num_dec_layers = class_preds.shape[1]

        for i in range(num_dec_layers):
          o_bbox = bbox_preds[:, i, :, :].sigmoid().to(device)
          o_cl = class_preds[:, i, :, :].to(device)

          for o_bbox_i, t_bbox, o_cl_i, t_cl, t_mask in zip(o_bbox, tgt_bbox, o_cl, tgt_cl, tgt_mask):
            loss_class, loss_bbox, loss_giou = compute_sample_loss(o_bbox_i, t_bbox, o_cl_i, t_cl, t_mask, empty_class_id=0, device=device)
            sample_loss = 1 * loss_class + 5 * loss_bbox + 2 * loss_giou
            loss += sample_loss / batch_size / num_dec_layers
            loss_class_batch += loss_class / batch_size / num_dec_layers
            loss_bbox_batch += loss_bbox / batch_size / num_dec_layers
            loss_giou_batch += loss_giou / batch_size / num_dec_layers

      optimizer.zero_grad()
      scaler.scale(loss).backward()
      scaler.unscale_(optimizer)
      nn.utils.clip_grad_norm_(model.parameters(), 0.1)
      scaler.step(optimizer)
      scaler.update()

      losses = torch.cat((losses, loss.detach().unsqueeze(0)))
      class_losses = torch.cat((class_losses, loss_class_batch.detach().unsqueeze(0)))
      box_losses = torch.cat((box_losses, loss_bbox_batch.detach().unsqueeze(0)))
      giou_losses = torch.cat((giou_losses, loss_giou_batch.detach().unsqueeze(0)))

    print(f"Total loss: {loss.item():.4f}; cls_loss: {loss_class_batch.item():.4f}; box_loss: {loss_bbox_batch.item():.4f}; giou_loss: {loss_giou_batch.item():.4f}")