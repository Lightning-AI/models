import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from litmodels import upload_model


def duplicate_hf_model(
    hf_model: str, lit_model: Optional[str] = None, local_workdir: Optional[str] = None, verbose: int = 1
) -> str:
    """Downloads the model from Hugging Face and uploads it to Lightning Cloud.

    Args:
        hf_model: The name of the Hugging Face model to duplicate.
        lit_model: The name of the Lightning Cloud model to create.
        local_workdir:
            The local working directory to use for the duplication process. If not set a temp folder will be created.
        verbose: Shot a progress bar for the upload.

    Returns:
        The name of the duplicated model in Lightning Cloud.
    """
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Hugging Face Hub is not installed. Please install it with `pip install huggingface_hub`."
        )

    if not local_workdir:
        local_workdir = tempfile.mkdtemp()
    local_workdir = Path(local_workdir)
    model_name = hf_model.replace("/", "_")

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    # Download the model from Hugging Face
    snapshot_download(
        repo_id=hf_model,
        revision="main",  # Branch/tag/commit
        repo_type="model",  # Options: "dataset", "model", "space"
        local_dir=local_workdir / model_name,  # Specify to save in custom location, default is cache
        local_dir_use_symlinks=True,  # Use symlinks to save disk space
        ignore_patterns=[".cache*"],  # Exclude certain files if needed
        max_workers=os.cpu_count(),  # Number of parallel downloads
    )
    # prune cache in the downloaded model
    for path in local_workdir.rglob(".cache*"):
        shutil.rmtree(path)

    # Upload the model to Lightning Cloud
    if not lit_model:
        lit_model = model_name
    model = upload_model(name=lit_model, model=local_workdir / model_name, verbose=verbose)
    return model.name
