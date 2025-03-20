from typing import TYPE_CHECKING, Any

from lightning_sdk.lightning_cloud.login import Auth
from lightning_utilities.core.rank_zero import rank_zero_only

from litmodels import upload_model
from litmodels.integrations.imports import _LIGHTNING_AVAILABLE, _PYTORCHLIGHTNING_AVAILABLE

if _LIGHTNING_AVAILABLE:
    from lightning.pytorch.callbacks import ModelCheckpoint as _LightningModelCheckpoint

    if TYPE_CHECKING:
        from lightning.pytorch import Trainer


if _PYTORCHLIGHTNING_AVAILABLE:
    from pytorch_lightning.callbacks import ModelCheckpoint as _PytorchLightningModelCheckpoint

    if TYPE_CHECKING:
        from pytorch_lightning import Trainer


# Base class to be inherited
class LitModelCheckpointMixin:
    """Mixin class for LitModel checkpoint functionality."""

    def __init__(self, model_name: str, *args: Any, **kwargs: Any) -> None:
        """Initialize with model name."""
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


# Create specific implementations
if _LIGHTNING_AVAILABLE:

    class LightningModelCheckpoint(LitModelCheckpointMixin, _LightningModelCheckpoint):
        """Lightning ModelCheckpoint with LitModel support.

        Args:
            model_name: Name of the model to upload in format 'organization/teamspace/modelname'
            args: Additional arguments to pass to the parent class.
            kwargs: Additional keyword arguments to pass to the parent class.
        """

        def __init__(self, model_name: str, *args: Any, **kwargs: Any) -> None:
            """Initialize the checkpoint with model name and other parameters."""
            _LightningModelCheckpoint.__init__(self, *args, **kwargs)
            LitModelCheckpointMixin.__init__(self, model_name)

        def _save_checkpoint(self, trainer: "Trainer", filepath: str) -> None:
            super()._save_checkpoint(trainer, filepath)
            if trainer.is_global_zero:
                # Only upload from the main process
                self._upload_model(filepath)


if _PYTORCHLIGHTNING_AVAILABLE:

    class PytorchLightningModelCheckpoint(LitModelCheckpointMixin, _PytorchLightningModelCheckpoint):
        """PyTorch Lightning ModelCheckpoint with LitModel support.

        Args:
            model_name: Name of the model to upload in format 'organization/teamspace/modelname'
            args: Additional arguments to pass to the parent class.
            kwargs: Additional keyword arguments to pass to the parent class.
        """

        def __init__(self, model_name: str, *args: Any, **kwargs: Any) -> None:
            """Initialize the checkpoint with model name and other parameters."""
            _PytorchLightningModelCheckpoint.__init__(self, *args, **kwargs)
            LitModelCheckpointMixin.__init__(self, model_name)

        def _save_checkpoint(self, trainer: "Trainer", filepath: str) -> None:
            super()._save_checkpoint(trainer, filepath)
            if trainer.is_global_zero:
                # Only upload from the main process
                self._upload_model(filepath)
