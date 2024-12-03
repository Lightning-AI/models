# Lightning Models

This package provides utilities for saving and loading machine learning models using PyTorch Lightning. It aims to simplify the process of managing model checkpoints, making it easier to save, load, and share models.

## Features

- **Save Models**: Easily save your trained models to cloud storage.
- **Load Models**: Load pre-trained models for inference or further training.
- **Checkpoint Management**: Manage multiple checkpoints with ease.
- **Cloud Integration**: Support for saving and loading models from cloud storage services.

[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://lightning.ai/)
[![CI testing](https://github.com/Lightning-AI/models/actions/workflows/ci-testing.yml/badge.svg?event=push)](https://github.com/Lightning-AI/models/actions/workflows/ci-testing.yml)
[![General checks](https://github.com/Lightning-AI/models/actions/workflows/ci-checks.yml/badge.svg?event=push)](https://github.com/Lightning-AI/models/actions/workflows/ci-checks.yml)
[![Documentation Status](https://readthedocs.org/projects/models/badge/?version=latest)](https://models.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Lightning-AI/models/main.svg?badge_token=mqheL1-cTn-280Vx4cJUdg)](https://results.pre-commit.ci/latest/github/Lightning-AI/models/main?badge_token=mqheL1-cTn-280Vx4cJUdg)

## Installation

To install the package, you can use `pip` from [Test PyPI](https://test.pypi.org/project/litmodels/):

```bash
pip install "litmodels==0.X.Y" --extra-index-url="https://test.pypi.org/simple/"
```

Or installing from source:

```bash
pip install https://github.com/Lightning-AI/models/archive/refs/heads/main.zip
```

## Usage

Here's a simple example of how to save and load a model using `litmodels`. First, you need to train a model using PyTorch Lightning. Then, you can save the model using the `upload_model` function.

```python
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from litmodels import upload_model
from litmodels.demos import BoringModel


class LitModel(BoringModel):
    def training_step(self, batch, batch_idx: int):
        loss = self.step(batch)
        # logging the computed loss
        self.log("train_loss", loss)
        return {"loss": loss}


# Define the model name - this should be unique to your model
# The format is <organization>/<teamspace>/<model-name>
MY_MODEL_NAME = "jirka/kaggle/lit-boring-model"

# Define the model
model = LitModel()
# Save the best model based on validation loss
checkpoint_callback = ModelCheckpoint(
    monitor="train_loss",  # Metric to monitor
    save_top_k=1,  # Only save the best model (use -1 to save all)
    mode="min",  # 'min' for loss, 'max' for accuracy
    save_last=True,  # Additionally save the last checkpoint
    dirpath="my_checkpoints/",  # Directory to save checkpoints
    filename="{epoch:02d}-{val_loss:.2f}",  # Custom checkpoint filename
)

# Train the model
trainer = Trainer(
    max_epochs=2,
    callbacks=[checkpoint_callback],
)
trainer.fit(model)

# Upload the best model to cloud storage
print(f"last: {vars(checkpoint_callback)}")
upload_model(model=checkpoint_callback.last_model_path, name=MY_MODEL_NAME)
```

To load the model, use the `load_model` function.

```python
from lightning import Trainer
from litmodels import download_model
from litmodels.demos import BoringModel


class LitModel(BoringModel):
    def training_step(self, batch, batch_idx: int):
        loss = self.step(batch)
        # logging the computed loss
        self.log("train_loss", loss)
        return {"loss": loss}


# Define the model name - this should be unique to your model
# The format is <organization>/<teamspace>/<model-name>:<model-version>
MY_MODEL_NAME = "jirka/kaggle/lit-boring-model:latest"

# Load the model from cloud storage
model_path = download_model(name=MY_MODEL_NAME, download_dir="my_models")
print(f"model: {model_path}")

# Train the model with extended training period
trainer = Trainer(max_epochs=4)
trainer.fit(
    LitModel(),
    ckpt_path=model_path,
)
```
