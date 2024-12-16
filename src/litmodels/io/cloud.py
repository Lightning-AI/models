# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
from typing import TYPE_CHECKING, Optional, Tuple, Union, List

from lightning_sdk.lightning_cloud.env import LIGHTNING_CLOUD_URL
from lightning_sdk.models import download_model, upload_model

if TYPE_CHECKING:
    from lightning_sdk.models import UploadedModelInfo

# if module_available("lightning"):
#     from lightning import LightningModule
# elif module_available("pytorch_lightning"):
#     from pytorch_lightning import LightningModule
# else:
#     LightningModule = None

_SHOWED_MODEL_LINKS = []


def _parse_name(name: str) -> Tuple[str, str, str]:
    """Parse the name argument into its components."""
    try:
        org_name, teamspace_name, model_name = name.split("/")
    except ValueError as err:
        raise ValueError(
            f"The name argument must be in the format 'organization/teamspace/model` but you provided '{name}'."
        ) from err
    return org_name, teamspace_name, model_name


def _print_model_link(name: str, verbose: Union[bool, int]) -> None:
    """Print a link to the uploaded model.

    Args:
        org_name: Name of the organization.
        teamspace_name: Name of the teamspace.
        model_name: Name of the model.
        verbose: Whether to print the link:

            - If set to 0, no link will be printed.
            - If set to 1, the link will be printed only once.
            - If set to 2, the link will be printed every time.
    """
    org_name, teamspace_name, model_name = _parse_name(name)
    url = f"{LIGHTNING_CLOUD_URL}/{org_name}/{teamspace_name}/models/{model_name}"
    msg = f"Model uploaded successfully. Link to the model: '{url}'"
    if int(verbose) > 1:
        print(msg)
    elif url not in _SHOWED_MODEL_LINKS:
        print(msg)
        _SHOWED_MODEL_LINKS.append(url)


def upload_model_files(
    name: str,
    path: str,
    progress_bar: bool = True,
    cloud_account: Optional[str] = None,
    verbose: Union[bool, int] = 1,
) -> "UploadedModelInfo":
    """Upload a local checkpoint file to the model store.

    Args:
        name: Name of the model to upload. Must be in the format 'organization/teamspace/modelname'
            where entity is either your username or the name of an organization you are part of.
        path: Path to the model file to upload.
        progress_bar: Whether to show a progress bar for the upload.
        cloud_account: The name of the cloud account to store the Model in. Only required if it can't be determined
            automatically.
        verbose: Whether to print a link to the uploaded model. If set to 0, no link will be printed.

    """
    info = upload_model(
        name=name,
        path=path,
        progress_bar=progress_bar,
        cloud_account=cloud_account,
    )
    if verbose:
        _print_model_link(info.name, verbose)
    return info


def download_model_files(
    name: str,
    download_dir: str = ".",
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
    return download_model(
        name=name,
        download_dir=download_dir,
        progress_bar=progress_bar,
    )
