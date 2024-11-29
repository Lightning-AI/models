# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Union

from lightning_sdk.api.teamspace_api import UploadedModelInfo
from lightning_sdk.teamspace import Teamspace
from lightning_sdk.utils import resolve as sdk_resolvers
from lightning_utilities import module_available

if TYPE_CHECKING:
    from torch.nn import Module

if module_available("torch"):
    import torch
    from torch.nn import Module
else:
    torch = None

# if module_available("lightning"):
#     from lightning import LightningModule
# elif module_available("pytorch_lightning"):
#     from pytorch_lightning import LightningModule
# else:
#     LightningModule = None


def _parse_name(name: str) -> Tuple[str, str, str]:
    """Parse the name argument into its components."""
    try:
        org_name, teamspace_name, model_name = name.split("/")
    except ValueError as err:
        raise ValueError(
            f"The name argument must be in the format 'organization/teamspace/model` but you provided '{name}'."
        ) from err
    return org_name, teamspace_name, model_name


def _get_teamspace(name: str, organization: str) -> Teamspace:
    """Get a Teamspace object from the SDK."""
    from lightning_sdk.api import OrgApi, UserApi

    org_api = OrgApi()
    user = sdk_resolvers._get_authed_user()
    teamspaces = {}
    for ts in UserApi()._get_all_teamspace_memberships(""):
        if ts.owner_type == "organization":
            org = org_api._get_org_by_id(ts.owner_id)
            teamspaces[f"{org.name}/{ts.name}"] = {"name": ts.name, "org": org.name}
        elif ts.owner_type == "user":  # todo: check also the name
            teamspaces[f"{user.name}/{ts.name}"] = {"name": ts.name, "user": user}
        else:
            raise RuntimeError(f"Unknown organization type {ts.organization_type}")

    requested_teamspace = f"{organization}/{name}".lower()
    if requested_teamspace not in teamspaces:
        options = "\n\t".join(teamspaces.keys())
        raise RuntimeError(f"Teamspace `{requested_teamspace}` not found. Available teamspaces: \n\t{options}")
    return Teamspace(**teamspaces[requested_teamspace])


def upload_model(
    model: Union[str, Path, "Module"],
    name: str,
    progress_bar: bool = True,
    cluster_id: Optional[str] = None,
    staging_dir: Optional[str] = None,
) -> UploadedModelInfo:
    """Upload a local checkpoint file to the model store.

    Args:
        model: The model to upload. Can be a path to a checkpoint file, a PyTorch model, or a Lightning model.
        name: Name tag of the model to upload. Must be in the format 'organization/teamspace/modelname'
            where entity is either your username or the name of an organization you are part of.
        progress_bar: Whether to show a progress bar for the upload.
        cluster_id: The name of the cluster to use. Only required if it can't be determined
            automatically.
        staging_dir: A directory where the model can be saved temporarily. If not provided, a temporary directory will
            be created and used.

    """
    if not staging_dir:
        staging_dir = tempfile.mkdtemp()
    # if LightningModule and isinstance(model, LightningModule):
    #     path = os.path.join(staging_dir, f"{model.__class__.__name__}.ckpt")
    #     model.save_checkpoint(path)
    elif torch and isinstance(model, Module):
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
    )


def upload_model_files(
    path: str,
    name: str,
    progress_bar: bool = True,
    cluster_id: Optional[str] = None,
) -> UploadedModelInfo:
    """Upload a local checkpoint file to the model store.

    Args:
        path: Path to the model file to upload.
        name: Name tag of the model to upload. Must be in the format 'organization/teamspace/modelname'
            where entity is either your username or the name of an organization you are part of.
        progress_bar: Whether to show a progress bar for the upload.
        cluster_id: The name of the cluster to use. Only required if it can't be determined
            automatically.

    """
    org_name, teamspace_name, model_name = _parse_name(name)
    teamspace = _get_teamspace(name=teamspace_name, organization=org_name)
    return teamspace.upload_model(
        path=path,
        name=model_name,
        progress_bar=progress_bar,
        cluster_id=cluster_id,
    )


def download_model_files(
    name: str,
    download_dir: str = ".",
    progress_bar: bool = True,
) -> str:
    """Download a checkpoint from the model store.

    Args:
        name: Name tag of the model to download. Must be in the format 'organization/teamspace/modelname'
            where entity is either your username or the name of an organization you are part of.
        download_dir: A path to directory where the model should be downloaded. Defaults
            to the current working directory.
        progress_bar: Whether to show a progress bar for the download.

    Returns:
        The absolute path to the downloaded model file or folder.
    """
    org_name, teamspace_name, model_name = _parse_name(name)
    teamspace = _get_teamspace(name=teamspace_name, organization=org_name)
    return teamspace.download_model(
        name=model_name,
        download_dir=download_dir,
        progress_bar=progress_bar,
    )
