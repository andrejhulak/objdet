import torch
import sys
# from utils.visualize import visualize_predictions
from utils.sparse_visualize import visualize_predictions
from dataset.VisDroneDS.utils import VisDrone_CLASS_NAMES

class BaseTrainer():
  '''
  This class is the base class for training objdet models
  '''
  def __init__(self, model:torch.nn, device:torch.device,
               train_dataloader:torch.utils.data.DataLoader,
               val_dataloader:torch.utils.data.DataLoader,
               epochs:int):
    '''
    Initialises the BaseTrainer class which sets up all the necessary variables
    and trains the model for a desired number of epochs (NOTE: it's very limited as of now)
    '''
    self.device = device if device is not None else "cuda" if torch.cuda.is_available else "cpu"
    self.model = model

    # data

    self.train_dataloader = train_dataloader
    self.val_dataloader = val_dataloader

    self.epochs = epochs

    # for initialising training
    self.amp = None
    self.scaler = None
    self.accumulate_loss = None
    self.lr = None
    self.optimizer = None

  def _init_training(self):
    self.model.to(self.device)

    #TODO change this whole thing to acutally init training properly
    self.amp = True #TODO: write an AMP checker self.amp = check_amp() -> bool

    self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

    self.accumulate_loss = self.val_dataloader.batch_size

    self.lr = 0.000025 #TODO write something to determine this automatically
    self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    self.model.train()

  # def _train_model(self):
  #   assert self.model.training

  #   for epoch in range(1, self.epochs + 1):
  #     self.epoch = epoch

  #     for i, batch in enumerate(self.train_dataloader):
  #       self.model.train()
  #       # torch.cuda.empty_cache()
  #       self.optimizer.zero_grad()

  #       with torch.autocast(device_type=self.device):
  #         batch = self.preprocess_batch(batch) 
  #         self.loss, self.loss_items = self.process_loss(self.model(batch["images"], batch["targets"]))

  #       self.scaler.scale(self.loss).backward()

  #       self.scaler.step(optimizer=self.optimizer)
  #       self.scaler.update()

  #       self.print_after_batch_results()
  #       if epoch % 100 == 0:
  #         visualize_predictions(self.model, self.train_dataloader, VisDrone_CLASS_NAMES)

  def _train_model(self):
    for epoch in range(1, self.epochs + 1):
      self.epoch = epoch
      for i, (images, target_tensor, batch_len, _) in enumerate(self.train_dataloader):
        self.model.train()
        self.optimizer.zero_grad()
        images = images.to(self.device)
        target_tensor = target_tensor.to(self.device)
        with torch.autocast(device_type=self.device, enabled=self.amp):
          out = self.model(images, targets={"target": target_tensor, "batch_len": batch_len})
          self.loss, self.loss_items = self.process_loss(out)
        self.scaler.scale(self.loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.print_after_batch_results()
        if epoch % 200 == 0:
          visualize_predictions(self.model, self.train_dataloader, VisDrone_CLASS_NAMES, self.device)

  def train(self):
    self._init_training()
    self._train_model()

  def preprocess_batch(self, batch):
    batch["images"] = [img.to(self.device) for img in batch["images"]]

    for target in batch["targets"]:
      target["boxes"] = target["boxes"].to(self.device)
      target["labels"] = target["labels"].to(self.device)

    return batch

  def process_loss(self, loss):
    loss["match_num"] = torch.tensor(0).to(self.device)
    total_loss = sum(loss.values())
    loss_items = torch.stack([loss[k] for k in loss.keys()]).detach().cpu()
    return total_loss, loss_items

  def print_after_batch_results(self):
    gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)

    batch_summary = (
      f"Loss: total={self.loss:.4f} "
      f"cls={self.loss_items[0]:.4f}, bbox={self.loss_items[1]:.4f}, "
      f"obj={self.loss_items[2]:.4f}, rpn={self.loss_items[3]:.4f}) | "
      f"Batch Size: {self.train_dataloader.batch_size} | "
      f"Epoch: {self.epoch} | "
      f"GPU Mem: {gpu_memory:.1f}GB"
    )

    print(batch_summary, end='\r')
    sys.stdout.flush()

  def evaluate(self):
    self.model.eval()
    pass