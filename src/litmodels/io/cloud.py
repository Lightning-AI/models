# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
from typing import Optional, Tuple, Union

from lightning_sdk.api.teamspace_api import UploadedModelInfo
from lightning_sdk.teamspace import Teamspace
from lightning_sdk.utils import resolve as sdk_resolvers

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


def _print_model_link(org_name: str, teamspace_name: str, model_name: str, verbose: Union[bool, int]) -> None:
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
    url = f"https://lightning.ai/{org_name}/{teamspace_name}/models/{model_name}"
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
    cluster_id: Optional[str] = None,
    verbose: Union[bool, int] = 0,
) -> UploadedModelInfo:
    """Upload a local checkpoint file to the model store.

    Args:
        name: Name of the model to upload. Must be in the format 'organization/teamspace/modelname'
            where entity is either your username or the name of an organization you are part of.
        path: Path to the model file to upload.
        progress_bar: Whether to show a progress bar for the upload.
        cluster_id: The name of the cluster to use. Only required if it can't be determined
            automatically.
        verbose: Whether to print a link to the uploaded model. If set to 0, no link will be printed.

    """
    org_name, teamspace_name, model_name = _parse_name(name)
    teamspace = _get_teamspace(name=teamspace_name, organization=org_name)
    info = teamspace.upload_model(
        path=path,
        name=model_name,
        progress_bar=progress_bar,
        cluster_id=cluster_id,
    )
    if verbose:
        _print_model_link(org_name, teamspace_name, model_name, verbose)
    return info


def download_model_files(
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
    org_name, teamspace_name, model_name = _parse_name(name)
    teamspace = _get_teamspace(name=teamspace_name, organization=org_name)
    return teamspace.download_model(
        name=model_name,
        download_dir=download_dir,
        progress_bar=progress_bar,
    )
