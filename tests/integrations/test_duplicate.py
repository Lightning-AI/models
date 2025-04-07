import os
from unittest import mock

from lightning_sdk.utils.resolve import _resolve_teamspace
from litmodels.integrations.duplicate import duplicate_hf_model

from tests.integrations import LIT_ORG, LIT_TEAMSPACE


@mock.patch("litmodels.integrations.duplicate.snapshot_download")
@mock.patch("litmodels.integrations.duplicate.upload_model_files")
def test_duplicate_hf_model(mock_upload_model, mock_snapshot_download, tmp_path):
    """Verify that the HF model can be duplicated to the teamspace"""

    # model name with random hash
    model_name = f"litmodels_hf_model+{os.urandom(8).hex()}"
    teamspace = _resolve_teamspace(org=LIT_ORG, teamspace=LIT_TEAMSPACE, user=None)
    org_team = f"{teamspace.owner.name}/{teamspace.name}"

    hf_model = "google/t5-efficient-tiny"
    duplicate_hf_model(hf_model=hf_model, lit_model=f"{org_team}/{model_name}", local_workdir=str(tmp_path))

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
        name=f"{org_team}/{model_name}",
        path=tmp_path / hf_model.replace("/", "_"),
        metadata={"hf_model": hf_model, "litModels_integration": "duplicate_hf_model"},
        verbose=1,
    )
