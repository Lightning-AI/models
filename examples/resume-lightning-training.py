"""
This example demonstrates how to resume training of a model using the `download_model` function.
"""

import os

from lightning import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from litmodels import download_model

# Define the model name - this should be unique to your model
# The format is <organization>/<teamspace>/<model-name>:<model-version>
MY_MODEL_NAME = "lightning-ai/jirka/lit-boring-callback:latest"


if __name__ == "__main__":
    model_files = download_model(name=MY_MODEL_NAME, download_dir="my_models")
    model_path = os.path.join("my_models", model_files[0])
    print(f"model: {model_path}")

    trainer = Trainer(max_epochs=4)
    trainer.fit(
        BoringModel(),
        ckpt_path=model_path,
    )
