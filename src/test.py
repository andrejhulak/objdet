from dataset.ArmaDS.ArmaDS import ArmaDS
from utils.visualize_batch_arma import visualize_batch
from torch.utils.data import DataLoader

if __name__ == "__main__":
  train_ds = ArmaDS(root="data/TEST_Arma/train")
  train_dl = DataLoader(dataset=train_ds, batch_size=1)
  visualize_batch(train_dl)