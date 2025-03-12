from typing import Any, Type, TypeVar, cast

from lightning.fabric.utilities import rank_zero_only
from lightning_sdk.lightning_cloud.login import Auth

from litmodels import upload_model
from litmodels.integrations.imports import _LIGHTNING_AVAILABLE, _PYTORCHLIGHTNING_AVAILABLE

if _LIGHTNING_AVAILABLE:
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint as LightningModelCheckpoint
if _PYTORCHLIGHTNING_AVAILABLE:
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint as PytorchLightningModelCheckpoint


# Type variable for the ModelCheckpoint class
ModelCheckpointType = TypeVar("ModelCheckpointType")


def _model_checkpoint_template(checkpoint_cls: Type[ModelCheckpointType]) -> Type[ModelCheckpointType]:
    """Template function that returns a LitModelCheckpoint class for a specific ModelCheckpoint class.

    Args:
        checkpoint_cls: The ModelCheckpoint class to extend

    Returns:
        A LitModelCheckpoint class extending the given ModelCheckpoint class
    """

    class LitModelCheckpointTemplate(checkpoint_cls):  # type: ignore
        """Lightning ModelCheckpoint with LitModel support.

        Args:
            model_name: Name of the model to upload. Must be in the format 'organization/teamspace/modelname'
                where entity is either your username or the name of an organization you are part of.
            args: Additional arguments to pass to the parent class.
            kwargs: Additional keyword arguments to pass to the parent class.
        """

        def __init__(self, model_name: str, *args: Any, **kwargs: Any) -> None:
            """Initialize the LitModelCheckpoint."""
            super().__init__(*args, **kwargs)
            self.model_name = model_name

            try:  # authenticate before anything else starts
                auth = Auth()
                auth.authenticate()
            except Exception:
                raise ConnectionError("Unable to authenticate with Lightning Cloud. Check your credentials.")

        @rank_zero_only
        def _upload_model(self, filepath: str) -> None:
            # todo: uploading on background so training does nt stops
            # todo: use filename as version but need to validate that such version does not exists yet
            upload_model(name=self.model_name, model=filepath)

        def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:
            super()._save_checkpoint(trainer, filepath)
            self._upload_model(filepath)

    return cast(Type[ModelCheckpointType], LitModelCheckpointTemplate)


# Create explicit classes with specific names
if _LIGHTNING_AVAILABLE:
    LightningModelCheckpoint = _model_checkpoint_template(LightningModelCheckpoint)
if _PYTORCHLIGHTNING_AVAILABLE:
    PTLightningModelCheckpoint = _model_checkpoint_template(PytorchLightningModelCheckpoint)
