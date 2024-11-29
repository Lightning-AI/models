import torch.utils.data as data
import torchvision as tv
from lightning import Callback, Trainer
from litmodels import upload_model_files
from sample_model import LitAutoEncoder


class UploadModelCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # Get the best model path from the checkpoint callback
        best_model_path = trainer.checkpoint_callback.best_model_path
        if best_model_path:
            print(f"Uploading model: {best_model_path}")
            upload_model_files(path=best_model_path, name="jirka/kaggle/lit-auto-encoder-callback")


if __name__ == "__main__":
    dataset = tv.datasets.MNIST(".", download=True, transform=tv.transforms.ToTensor())
    train, val = data.random_split(dataset, [55000, 5000])

    autoencoder = LitAutoEncoder()

    trainer = Trainer(
        max_epochs=2,
        callbacks=[UploadModelCallback()],
    )
    trainer.fit(
        autoencoder,
        data.DataLoader(train, batch_size=256),
        data.DataLoader(val, batch_size=256),
    )
