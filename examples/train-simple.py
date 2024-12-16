import torch.utils.data as data
import torchvision as tv
from lightning import Trainer
from litmodels import upload_model
from sample_model import LitAutoEncoder

# Define the model name - this should be unique to your model
# The format is <organization>/<teamspace>/<model-name>
MY_MODEL_NAME = "jirka/kaggle/lit-auto-encoder-simple"


if __name__ == "__main__":
    dataset = tv.datasets.MNIST(".", download=True, transform=tv.transforms.ToTensor())
    train, val = data.random_split(dataset, [55000, 5000])

    autoencoder = LitAutoEncoder()

    trainer = Trainer(max_epochs=2)
    trainer.fit(
        autoencoder,
        data.DataLoader(train, batch_size=256),
        data.DataLoader(val, batch_size=256),
    )
    checkpoint_path = getattr(trainer.checkpoint_callback, "best_model_path")
    print(f"best: {checkpoint_path}")
    upload_model(model=checkpoint_path, name=MY_MODEL_NAME)
