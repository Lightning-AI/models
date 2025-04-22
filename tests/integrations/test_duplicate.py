import os
from unittest import mock

from litmodels.integrations.duplicate import duplicate_hf_model


@mock.patch("litmodels.integrations.duplicate.snapshot_download")
@mock.patch("litmodels.integrations.duplicate.upload_model_files")
def test_duplicate_hf_model(mock_upload_model, mock_snapshot_download, tmp_path):
    """Verify that the HF model can be duplicated to the teamspace"""

    hf_model = "google/t5-efficient-tiny"
    # model name with random hash
    model_name = f"litmodels_hf_model+{os.urandom(8).hex()}"
    duplicate_hf_model(hf_model=hf_model, lit_model=model_name, local_workdir=str(tmp_path))

    mock_snapshot_download.assert_called_with(
        repo_id=hf_model,
        revision="main",
        repo_type="model",
        local_dir=tmp_path / hf_model.replace("/", "_"),
        local_dir_use_symlinks=True,
        ignore_patterns=[".cache*"],
        max_workers=os.cpu_count(),
    )
    mock_upload_model.assert_called_with(
        name=f"{model_name}",
        path=tmp_path / hf_model.replace("/", "_"),
        metadata={"hf_model": hf_model, "litModels.integration": "duplicate_hf_model"},
        verbose=1,
    )
