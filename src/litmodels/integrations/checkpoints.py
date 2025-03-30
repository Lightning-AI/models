from abc import ABC
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from lightning_sdk.lightning_cloud.login import Auth
from lightning_utilities.core.rank_zero import rank_zero_only, rank_zero_warn

from litmodels import upload_model
from litmodels.integrations.imports import _LIGHTNING_AVAILABLE, _PYTORCHLIGHTNING_AVAILABLE

if _LIGHTNING_AVAILABLE:
    from lightning.pytorch.callbacks import ModelCheckpoint as _LightningModelCheckpoint


if _PYTORCHLIGHTNING_AVAILABLE:
    from pytorch_lightning.callbacks import ModelCheckpoint as _PytorchLightningModelCheckpoint


if TYPE_CHECKING:
    if _LIGHTNING_AVAILABLE or _PYTORCHLIGHTNING_AVAILABLE:
        import lightning.pytorch as pl


# Base class to be inherited
class LitModelCheckpointMixin(ABC):
    """Mixin class for LitModel checkpoint functionality."""

    # mainly ofr mocking reasons
    _datetime_stamp: str = datetime.now().strftime("%Y%m%d-%H%M")

    def __init__(self, model_name: Optional[str]) -> None:
        """Initialize with model name."""
        if not model_name:
            rank_zero_warn(
                "The model is not defined so we will continue with LightningModule names and timestamp of now"
            )
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

    def _update_model_name(self, pl_model: "pl.LightningModule") -> None:
        if self.model_name:
            return
        # setting the model name as Lightning module with some time hash
        self.model_name = pl_model.__class__.__name__ + f"_{self._datetime_stamp}"


# Create specific implementations
if _LIGHTNING_AVAILABLE:

    class LightningModelCheckpoint(LitModelCheckpointMixin, _LightningModelCheckpoint):
        """Lightning ModelCheckpoint with LitModel support.

        Args:
            model_name: Name of the model to upload in format 'organization/teamspace/modelname'
            args: Additional arguments to pass to the parent class.
            kwargs: Additional keyword arguments to pass to the parent class.
        """

        def __init__(self, *args: Any, model_name: Optional[str] = None, **kwargs: Any) -> None:
            """Initialize the checkpoint with model name and other parameters."""
            _LightningModelCheckpoint.__init__(self, *args, **kwargs)
            LitModelCheckpointMixin.__init__(self, model_name)

        def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
            super().setup(trainer, pl_module, stage)
            self._update_model_name(pl_module)

        def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
            super()._save_checkpoint(trainer, filepath)
            if trainer.is_global_zero:  # Only upload from the main process
                self._upload_model(filepath)


if _PYTORCHLIGHTNING_AVAILABLE:

    class PytorchLightningModelCheckpoint(LitModelCheckpointMixin, _PytorchLightningModelCheckpoint):
        """PyTorch Lightning ModelCheckpoint with LitModel support.

        Args:
            model_name: Name of the model to upload in format 'organization/teamspace/modelname'
            args: Additional arguments to pass to the parent class.
            kwargs: Additional keyword arguments to pass to the parent class.
        """

        def __init__(self, *args: Any, model_name: Optional[str] = None, **kwargs: Any) -> None:
            """Initialize the checkpoint with model name and other parameters."""
            _PytorchLightningModelCheckpoint.__init__(self, *args, **kwargs)
            LitModelCheckpointMixin.__init__(self, model_name)

        def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
            super().setup(trainer, pl_module, stage)
            self._update_model_name(pl_module)

        def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
            super()._save_checkpoint(trainer, filepath)
            if trainer.is_global_zero:  # Only upload from the main process
                self._upload_model(filepath)
