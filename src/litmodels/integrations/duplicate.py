import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from lightning_utilities import module_available

from litmodels.io import upload_model_files

if module_available("huggingface_hub"):
    from huggingface_hub import snapshot_download
else:
    snapshot_download = None


def duplicate_hf_model(
    hf_model: str,
    lit_model: Optional[str] = None,
    local_workdir: Optional[str] = None,
    verbose: int = 1,
    metadata: Optional[dict] = None,
) -> str:
    """Downloads the model from Hugging Face and uploads it to Lightning Cloud.

    Args:
        hf_model: The name of the Hugging Face model to duplicate.
        lit_model: The name of the Lightning Cloud model to create.
        local_workdir:
            The local working directory to use for the duplication process. If not set a temp folder will be created.
        verbose: Shot a progress bar for the upload.
        metadata: Optional metadata to attach to the model. If not provided, a default metadata will be used.

    Returns:
        The name of the duplicated model in Lightning Cloud.
    """
    if not snapshot_download:
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
    if not metadata:
        metadata = {}
    metadata.update({"litModels_integration": "duplicate_hf_model", "hf_model": hf_model})
    model = upload_model_files(name=lit_model, path=local_workdir / model_name, verbose=verbose, metadata=metadata)
    return model.name
