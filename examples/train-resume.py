import torch.utils.data as data
import torchvision as tv
from lightning import Trainer
from litmodels import download_model_files
from sample_model import LitAutoEncoder

if __name__ == "__main__":
    dataset = tv.datasets.MNIST(".", download=True, transform=tv.transforms.ToTensor())
    train, val = data.random_split(dataset, [55000, 5000])

    model_path = download_model_files(name="jirka/kaggle/lit-auto-encoder-simple", download_dir="my_models")
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
