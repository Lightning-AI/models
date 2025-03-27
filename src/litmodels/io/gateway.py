import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Union

import joblib
from lightning_utilities import module_available

from litmodels.io.cloud import download_model_files, upload_model_files

if module_available("torch"):
    import torch
else:
    torch = None

if TYPE_CHECKING:
    from lightning_sdk.models import UploadedModelInfo


def upload_model(
    name: str,
    model: Union[str, Path, "torch.nn.Module", Any],
    progress_bar: bool = True,
    cloud_account: Optional[str] = None,
    staging_dir: Optional[str] = None,
    verbose: Union[bool, int] = 1,
) -> "UploadedModelInfo":
    """Upload a checkpoint to the model store.

    Args:
        name: Name of the model to upload. Must be in the format 'organization/teamspace/modelname'
            where entity is either your username or the name of an organization you are part of.
        model: The model to upload. Can be a path to a checkpoint file, a PyTorch model, or a Lightning model.
        progress_bar: Whether to show a progress bar for the upload.
        cloud_account: The name of the cloud account to store the Model in. Only required if it can't be determined
            automatically.
        staging_dir: A directory where the model can be saved temporarily. If not provided, a temporary directory will
            be created and used.
        verbose: Whether to print some additional information about the uploaded model.

    """
    if not staging_dir:
        staging_dir = tempfile.mkdtemp()
    if isinstance(model, (str, Path)):
        path = model
    # if LightningModule and isinstance(model, LightningModule):
    #     path = os.path.join(staging_dir, f"{model.__class__.__name__}.ckpt")
    #     model.save_checkpoint(path)
    elif torch and isinstance(model, torch.jit.ScriptModule):
        path = os.path.join(staging_dir, f"{model.__class__.__name__}.ts")
        model.save(path)
    elif torch and isinstance(model, torch.nn.Module):
        path = os.path.join(staging_dir, f"{model.__class__.__name__}.pth")
        torch.save(model.state_dict(), path)
    else:
        path = os.path.join(staging_dir, f"{model.__class__.__name__}.pkl")
        joblib.dump(model, path)

    return upload_model_files(
        path=path,
        name=name,
        progress_bar=progress_bar,
        cloud_account=cloud_account,
        verbose=verbose,
    )


def download_model(
    name: str,
    download_dir: Union[str, Path] = ".",
    progress_bar: bool = True,
) -> Union[str, List[str]]:
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


def load_model(name: str, download_dir: str = ".") -> Any:
    """Download a model from the model store and load it into memory.

    Args:
        name: Name of the model to download. Must be in the format 'organization/teamspace/modelname'
            where entity is either your username or the name of an organization you are part of.
        download_dir: A path to directory where the model should be downloaded. Defaults
            to the current working directory.

    Returns:
        The loaded model.
    """
    download_paths = download_model(name=name, download_dir=download_dir)
    # filter out all Markdown, TXT and RST files
    download_paths = [p for p in download_paths if Path(p).suffix.lower() not in {".md", ".txt", ".rst"}]
    if len(download_paths) > 1:
        raise NotImplementedError("Downloaded model with multiple files is not supported yet.")
    model_path = Path(download_dir) / download_paths[0]
    if model_path.suffix.lower() == ".pkl":
        return joblib.load(model_path)
    if model_path.suffix.lower() == ".ts":
        return torch.jit.load(model_path)
    raise NotImplementedError(f"Loading model from {model_path.suffix} is not supported yet.")
