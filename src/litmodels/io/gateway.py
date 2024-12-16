import os
import tempfile
from pathlib import Path
from typing import Optional, Union

from lightning_sdk.models import UploadedModelInfo
from lightning_utilities import module_available

from litmodels.io.cloud import download_model_files, upload_model_files

if module_available("torch"):
    import torch
    from torch.nn import Module
else:
    torch = None


def upload_model(
    name: str,
    model: Union[str, Path, "Module"],
    progress_bar: bool = True,
    cluster_id: Optional[str] = None,
    staging_dir: Optional[str] = None,
    verbose: Union[bool, int] = 1,
) -> UploadedModelInfo:
    """Upload a checkpoint to the model store.

    Args:
        name: Name of the model to upload. Must be in the format 'organization/teamspace/modelname'
            where entity is either your username or the name of an organization you are part of.
        model: The model to upload. Can be a path to a checkpoint file, a PyTorch model, or a Lightning model.
        progress_bar: Whether to show a progress bar for the upload.
        cluster_id: The name of the cluster to use. Only required if it can't be determined
            automatically.
        staging_dir: A directory where the model can be saved temporarily. If not provided, a temporary directory will
            be created and used.
        verbose: Whether to print some additional information about the uploaded model.

    """
    if not staging_dir:
        staging_dir = tempfile.mkdtemp()
    # if LightningModule and isinstance(model, LightningModule):
    #     path = os.path.join(staging_dir, f"{model.__class__.__name__}.ckpt")
    #     model.save_checkpoint(path)
    if torch and isinstance(model, Module):
        path = os.path.join(staging_dir, f"{model.__class__.__name__}.pth")
        torch.save(model.state_dict(), path)
    elif isinstance(model, str):
        path = model
    elif isinstance(model, Path):
        path = str(model)
    else:
        raise ValueError(f"Unsupported model type {type(model)}")
    return upload_model_files(
        path=path,
        name=name,
        progress_bar=progress_bar,
        cluster_id=cluster_id,
        verbose=verbose,
    )


def download_model(
    name: str,
    download_dir: str = ".",
    progress_bar: bool = True,
) -> str:
    """Download a checkpoint from the model store.

    Args:
        name: Name of the model to download. Must be in the format 'organization/teamspace/modelname'
            where entity is either your username or the name of an organization you are part of.
        download_dir: A path to directory where the model should be downloaded. Defaults
            to the current working directory.
        progress_bar: Whether to show a progress bar for the download.

    Returns:
        The absolute path to the downloaded model file or folder.
    """
    return download_model_files(
        name=name,
        download_dir=download_dir,
        progress_bar=progress_bar,
    )
