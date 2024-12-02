import torch.utils.data as data
import torchvision as tv
from lightning import Callback, Trainer
from litmodels import upload_model
from sample_model import LitAutoEncoder

# Define the model name - this should be unique to your model
# The format is <organization>/<teamspace>/<model-name>
MY_MODEL_NAME = "jirka/kaggle/lit-auto-encoder-callback"


class UploadModelCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # Get the best model path from the checkpoint callback
        best_model_path = trainer.checkpoint_callback.best_model_path
        if best_model_path:
            print(f"Uploading model: {best_model_path}")
            upload_model(path=best_model_path, name=MY_MODEL_NAME)


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
