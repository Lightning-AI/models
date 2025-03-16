"""
This example demonstrates how to train a model and upload it to the cloud using the `upload_model` function.
"""

from lightning import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from litmodels import upload_model

# Define the model name - this should be unique to your model
# The format is <organization>/<teamspace>/<model-name>
MY_MODEL_NAME = "jirka/kaggle/lit-auto-encoder-simple"


if __name__ == "__main__":
    trainer = Trainer(max_epochs=2)
    trainer.fit(BoringModel())
    checkpoint_path = getattr(trainer.checkpoint_callback, "best_model_path")
    print(f"best: {checkpoint_path}")
    upload_model(model=checkpoint_path, name=MY_MODEL_NAME)
