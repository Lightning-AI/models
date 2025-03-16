"""
# Enhanced Logging with LightningLogger

Integrate with [LitLogger](https://github.com/gridai/lit-logger) to automatically log your model checkpoints
 and training metrics to cloud storage.
Though the example utilizes PyTorch Lightning, this integration concept works across various model training frameworks.

"""

from lightning import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from litlogger import LightningLogger


class DemoModel(BoringModel):
    def training_step(self, batch, batch_idx):
        output = super().training_step(batch, batch_idx)
        self.log("train_loss", output["loss"])
        return output


if __name__ == "__main__":
    # configure the logger
    lit_logger = LightningLogger(log_model=True)

    # pass logger to the Trainer
    trainer = Trainer(max_epochs=5, logger=lit_logger)

    # train the model
    trainer.fit(model=DemoModel())
