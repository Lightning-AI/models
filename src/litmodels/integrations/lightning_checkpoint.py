from litmodels import upload_model
from litmodels.integrations.imports import _LIGHTNING_AVAILABLE, _PYTORCHLIGHTNING_AVAILABLE

if _LIGHTNING_AVAILABLE:
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint
elif _PYTORCHLIGHTNING_AVAILABLE:
    from pytorch_lightning.callbacks import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
else:
    raise ModuleNotFoundError("No module named 'lightning' or 'pytorch_lightning'")


class LitModelCheckpoint(ModelCheckpoint):
    """Lightning ModelCheckpoint with LitModel support.

    Args:
        model_name: Name of the model to upload. Must be in the format 'organization/teamspace/modelname'
            where entity is either your username or the name of an organization you are part of.
        args: Additional arguments to pass to the parent class.
        kwargs: Additional keyword arguments to pass to the parent class.

    """

    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name

    def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        # todo: uploading on background so training does nt stops
        # todo: use filename as version but need to validate that such version does not exists yet
        upload_model(name=self.model_name, model=filepath)
