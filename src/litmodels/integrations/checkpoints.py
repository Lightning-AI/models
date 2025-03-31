from abc import ABC
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from lightning_sdk.lightning_cloud.login import Auth
from lightning_sdk.utils.resolve import _resolve_teamspace
from lightning_utilities.core.rank_zero import rank_zero_only, rank_zero_warn

from litmodels import upload_model
from litmodels.integrations.imports import _LIGHTNING_AVAILABLE, _PYTORCHLIGHTNING_AVAILABLE
from litmodels.io.cloud import _list_available_teamspaces

if _LIGHTNING_AVAILABLE:
    from lightning.pytorch.callbacks import ModelCheckpoint as _LightningModelCheckpoint


if _PYTORCHLIGHTNING_AVAILABLE:
    from pytorch_lightning.callbacks import ModelCheckpoint as _PytorchLightningModelCheckpoint


if TYPE_CHECKING:
    if _LIGHTNING_AVAILABLE:
        import lightning.pytorch as pl
    if _PYTORCHLIGHTNING_AVAILABLE:
        import pytorch_lightning as pl


# Base class to be inherited
class LitModelCheckpointMixin(ABC):
    """Mixin class for LitModel checkpoint functionality."""

    _auth: Auth  # preserve the auth object for later use
    # mainly ofr mocking reasons
    _datetime_stamp: str = datetime.now().strftime("%Y%m%d-%H%M")
    model_name: Optional[str] = None

    def __init__(self, model_name: Optional[str]) -> None:
        """Initialize with model name."""
        if not model_name:
            rank_zero_warn(
                "The model is not defined so we will continue with LightningModule names and timestamp of now"
            )
        # remove any / from beginning and end of the name
        self.model_name = model_name.strip("/") if model_name else None

        try:  # authenticate before anything else starts
            self._auth = Auth()
            self._auth.authenticate()
        except Exception:
            raise ConnectionError("Unable to authenticate with Lightning Cloud. Check your credentials.")

    @rank_zero_only
    def _upload_model(self, filepath: str) -> None:
        # todo: uploading on background so training does nt stops
        # todo: use filename as version but need to validate that such version does not exists yet
        if not self.model_name:
            raise RuntimeError(
                "Model name is not specified neither updated by `setup` method via Trainer."
                " Please set the model name before uploading or ensure that `setup` method is called."
            )
        upload_model(name=self.model_name, model=filepath)

    def _update_model_name(self, pl_model: "pl.LightningModule") -> None:
        """Update the model name if not already set."""
        count_slashes_in_name = self.model_name.count("/") if self.model_name else 0
        default_model_name = pl_model.__class__.__name__ + f"_{self._datetime_stamp}"
        if count_slashes_in_name > 2:
            raise ValueError(
                f"Invalid model name: '{self.model_name}'. It should not contain more than two '/' character."
            )
        elif count_slashes_in_name == 2:
            # user has defined the model name in the format 'organization/teamspace/modelname'
            return
        elif count_slashes_in_name == 1:
            # user had defined only the teamspace name
            self.model_name = f"{self.model_name}/{default_model_name}"
        elif count_slashes_in_name == 0:
            if not self.model_name:
                self.model_name = default_model_name
            teamspace = _resolve_teamspace(None, None, None)
            if teamspace:
                # case you use default model name and teamspace determined from env. variables aka running in studio
                self.model_name = default_model_name
            else:  # try to load default users teamspace
                ts_names = list(_list_available_teamspaces().keys())
                if len(ts_names) == 1:
                    self.model_name = f"{ts_names[0]}/{default_model_name}"
                else:
                    options = "\n\t".join(ts_names)
                    raise RuntimeError(
                        f"Teamspace is not defined and there are multiple teamspaces available:\n{options}"
                    )
        else:
            raise RuntimeError(f"Invalid model name: '{self.model_name}'")


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
            """Setup the checkpoint callback."""
            super().setup(trainer, pl_module, stage)
            self._update_model_name(pl_module)

        def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
            """Extend the save checkpoint method to upload the model."""
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
            """Setup the checkpoint callback."""
            super().setup(trainer, pl_module, stage)
            self._update_model_name(pl_module)

        def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
            """Extend the save checkpoint method to upload the model."""
            super()._save_checkpoint(trainer, filepath)
            if trainer.is_global_zero:  # Only upload from the main process
                self._upload_model(filepath)
