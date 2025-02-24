# Effortless Model Management for Your Models ⚡

__Effortless management for your ML models.__

<div align="center">

🚀 [Quick start](#quick-start)
📦 [Examples](#saving-and-loading-models)
📚 [Documentation](https://lightning.ai/docs/overview/model-registry)
💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)
📋 [License: Apache 2.0](https://github.com/Lightning-AI/models/blob/main/LICENSE)

</div>

**Lightning Models** is a streamlined toolkit for effortlessly saving, loading, and managing your model checkpoints. Designed to simplify the entire model lifecycle—from training and inference to sharing, deployment, and cloud integration—Lightning Models supports any framework that produces model checkpoints, including but not limited to PyTorch Lightning.

<pre>
✅ Seamless Model Saving & Loading
✅ Robust Checkpoint Management
✅ Cloud Integration Out of the Box
✅ Versatile Across Frameworks
</pre>

# Quick start

Install Lightning Models via pip (more installation options below):

```bash
pip install -U litmodels
```

Or install directly from source:

```bash
pip install https://github.com/Lightning-AI/models/archive/refs/heads/main.zip
```

## Saving and Loading Models

Lightning Models offers a simple API to manage your model checkpoints.
Train your model using your preferred framework (our examples demonstrate PyTorch Lightning integration) and then save your best checkpoint with a single function call.

### Saving a Model

```python
from lightning import Trainer
from litmodels import upload_model
from litmodels.demos import BoringModel

# Define the model name - this should be unique to your model
MY_MODEL_NAME = "<organization>/<teamspace>/<model-name>"


class LitModel(BoringModel):
    def training_step(self, batch, batch_idx: int):
        loss = self.step(batch)
        # logging the computed loss
        self.log("train_loss", loss)
        return {"loss": loss}


# Configure Lightning Trainer
trainer = Trainer(max_epochs=2)
# Define the model and train it
trainer.fit(LitModel())

# Upload the best model to cloud storage
checkpoint_path = getattr(trainer.checkpoint_callback, "best_model_path")
upload_model(model=checkpoint_path, name=MY_MODEL_NAME)
```

### Loading a Model

```python
from lightning import Trainer
from litmodels import download_model
from litmodels.demos import BoringModel

# Define the model name - this should be unique to your model
MY_MODEL_NAME = "<organization>/<teamspace>/<model-name>:<model-version>"


class LitModel(BoringModel):
    def training_step(self, batch, batch_idx: int):
        loss = self.step(batch)
        # logging the computed loss
        self.log("train_loss", loss)
        return {"loss": loss}


# Load the model from cloud storage
checkpoint_path = download_model(name=MY_MODEL_NAME, download_dir="my_models")
print(f"model: {checkpoint_path}")

# Train the model with extended training period
trainer = Trainer(max_epochs=4)
trainer.fit(LitModel(), ckpt_path=checkpoint_path)
```

## Advanced Checkpointing Workflow

Enhance your training process with an automatic checkpointing callback that uploads the best model at the end of each epoch.
While the example uses PyTorch Lightning callbacks, similar workflows can be implemented in any training loop that produces checkpoints.

```python
import os
import torch.utils.data as data
import torchvision as tv
from lightning import Callback, Trainer
from litmodels import upload_model
from litmodels.demos import BoringModel

# Define the model name - this should be unique to your model
MY_MODEL_NAME = "<organization>/<teamspace>/<model-name>"


class LitModel(BoringModel):
    def training_step(self, batch, batch_idx: int):
        loss = self.step(batch)
        # logging the computed loss
        self.log("train_loss", loss)
        return {"loss": loss}


class UploadModelCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # Get the best model path from the checkpoint callback
        checkpoint_path = getattr(trainer.checkpoint_callback, "best_model_path")
        if checkpoint_path and os.path.exists(checkpoint_path):
            upload_model(model=checkpoint_path, name=MY_MODEL_NAME)


dataset = tv.datasets.MNIST(".", download=True, transform=tv.transforms.ToTensor())
train, val = data.random_split(dataset, [55000, 5000])

trainer = Trainer(
    max_epochs=2,
    callbacks=[UploadModelCallback()],
)
trainer.fit(
    LitModel(),
    data.DataLoader(train, batch_size=256),
    data.DataLoader(val, batch_size=256),
)
```

## Enhanced Logging with LightningLogger

Integrate with [LitLogger](https://github.com/gridai/lit-logger) to automatically log your model checkpoints and training metrics to cloud storage.
Though the example utilizes PyTorch Lightning, this integration concept works across various model training frameworks.

```python
import os
import lightning as L
from psutil import cpu_count
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from litlogger import LightningLogger


class LitAutoEncoder(L.LightningModule):

    def __init__(self, lr=1e-3, inp_size=28):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(inp_size * inp_size, 64), nn.ReLU(), nn.Linear(64, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, inp_size * inp_size)
        )
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
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


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
    trainer = L.Trainer(max_epochs=5, logger=lit_logger)

    # train the model
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)
```
