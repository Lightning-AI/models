"""
# Enhanced Logging with LightningLogger

Integrate with [LitLogger](https://github.com/gridai/lit-logger) to automatically log your model checkpoints
 and training metrics to cloud storage.
Though the example utilizes PyTorch Lightning, this integration concept works across various model training frameworks.

"""

import os

from lightning import LightningModule, Trainer
from litlogger import LightningLogger
from psutil import cpu_count
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class LitAutoEncoder(LightningModule):
    def __init__(self, lr=1e-3, inp_size=28):
        super().__init__()

        self.encoder = nn.Sequential(nn.Linear(inp_size * inp_size, 64), nn.ReLU(), nn.Linear(64, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, inp_size * inp_size))
        self.lr = lr
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # log metrics
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    # init the autoencoder
    autoencoder = LitAutoEncoder(lr=1e-3, inp_size=28)

    # setup data
    train_loader = DataLoader(
        dataset=MNIST(os.getcwd(), download=True, transform=ToTensor()),
        batch_size=32,
        shuffle=True,
        num_workers=cpu_count(),
        persistent_workers=True,
    )

    # configure the logger
    lit_logger = LightningLogger(log_model=True)

    # pass logger to the Trainer
    trainer = Trainer(max_epochs=5, logger=lit_logger)

    # train the model
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)
