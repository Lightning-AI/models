import torch.utils.data as data
import torchvision as tv
from lightning import Trainer
from litmodels import download_model
from sample_model import LitAutoEncoder

# Define the model name - this should be unique to your model
# The format is <organization>/<teamspace>/<model-name>:<model-version>
MY_MODEL_NAME = "jirka/kaggle/lit-auto-encoder-callback:latest"


if __name__ == "__main__":
    dataset = tv.datasets.MNIST(".", download=True, transform=tv.transforms.ToTensor())
    train, val = data.random_split(dataset, [55000, 5000])

    model_path = download_model(name=MY_MODEL_NAME, download_dir="my_models")
    print(f"model: {model_path}")
    # autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint_path=model_path)

    trainer = Trainer(
        max_epochs=4,
    )
    trainer.fit(
        LitAutoEncoder(),
        data.DataLoader(train, batch_size=256),
        data.DataLoader(val, batch_size=256),
        ckpt_path=model_path,
    )
