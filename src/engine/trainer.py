import torch

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
    self.model = model.to(device)

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
    #TODO change this whole thing to acutally init training properly
    self.amp = True #TODO: write an AMP checker self.amp = check_amp() -> bool

    self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
    self.accumulate_loss = self.val_dataloader.batch_size
    self.lr = 0.001 #TODO write something to determine this automatically
    self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    self.model.train()

  def _train_model(self):
    assert self.model.training
    for epoch in range(self.epochs):
      for i, batch in enumerate(self.train_dataloader):
        # delete cache
        torch.cuda.empty_cache()

        with torch.autocast(device_type=self.device):
          batch = self.preprocess_batch(batch) 
          out = self.model(batch["images"], batch["targets"])

  def train(self):
    self._init_training()
    self._train_model()

  def preprocess_batch(self, batch):
    batch["images"] = [img.to(self.device) for img in batch["images"]]

    for target in batch["targets"]:
      target["boxes"] = target["boxes"].to(self.device)
      target["labels"] = target["labels"].to(self.device)

    return batch