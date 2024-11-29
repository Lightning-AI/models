import torch.utils.data as data
import torchvision as tv
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from litmodels import upload_model_files
from sample_model import LitAutoEncoder

if __name__ == "__main__":
    dataset = tv.datasets.MNIST(".", download=True, transform=tv.transforms.ToTensor())
    train, val = data.random_split(dataset, [55000, 5000])

    autoencoder = LitAutoEncoder()
    # Save the best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # Metric to monitor
        save_top_k=1,  # Only save the best model (use -1 to save all)
        mode="min",  # 'min' for loss, 'max' for accuracy
        save_last=True,  # Additionally save the last checkpoint
        dirpath="my_checkpoints/",  # Directory to save checkpoints
        filename="{epoch:02d}-{val_loss:.2f}",  # Custom checkpoint filename
    )

    trainer = Trainer(
        max_epochs=2,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        autoencoder,
        data.DataLoader(train, batch_size=256),
        data.DataLoader(val, batch_size=256),
    )
    print(f"last: {vars(checkpoint_callback)}")
    upload_model_files(path=checkpoint_callback.last_model_path, name="jirka/kaggle/lit-auto-encoder-simple")
