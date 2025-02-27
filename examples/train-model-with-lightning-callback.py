"""
Train a model with a Lightning callback that uploads the best model to the cloud after each epoch.
"""

import torch.utils.data as data
import torchvision as tv
from lightning import Trainer
from litmodels.integrations.lightning_checkpoint import LitModelCheckpoint
from sample_model import LitAutoEncoder

# Define the model name - this should be unique to your model
# The format is <organization>/<teamspace>/<model-name>
MY_MODEL_NAME = "lightning-ai/jirka/lit-auto-encoder-callback"


if __name__ == "__main__":
    dataset = tv.datasets.MNIST(".", download=True, transform=tv.transforms.ToTensor())
    train, val = data.random_split(dataset, [55000, 5000])

    autoencoder = LitAutoEncoder()

    trainer = Trainer(
        max_epochs=2,
        callbacks=LitModelCheckpoint(model_name=MY_MODEL_NAME),
    )
    trainer.fit(
        autoencoder,
        data.DataLoader(train, batch_size=256),
        data.DataLoader(val, batch_size=256),
    )
