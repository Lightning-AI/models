"""
Train a model with a Lightning callback that uploads the best model to the cloud after each epoch.
"""

from lightning import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel

from litmodels.integrations import LightningModelCheckpoint

# Define the model name - this should be unique to your model
# The format is <organization>/<teamspace>/<model-name>
MY_MODEL_NAME = "lightning-ai/jirka/lit-boring-callback"


if __name__ == "__main__":
    trainer = Trainer(
        max_epochs=2,
        callbacks=LightningModelCheckpoint(model_registry=MY_MODEL_NAME),
    )
    trainer.fit(BoringModel())
