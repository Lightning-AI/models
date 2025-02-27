import os
from unittest import mock

import pytest
from litmodels import download_model, upload_model
from litmodels.io import upload_model_files
from sklearn import svm
from torch.nn import Module


@pytest.mark.parametrize("name", ["org/model", "model-name", "/too/many/slashes"])
def test_wrong_model_name(name):
    with pytest.raises(ValueError, match=r".*organization/teamspace/model.*"):
        upload_model_files(path="path/to/checkpoint", name=name)
    with pytest.raises(ValueError, match=r".*organization/teamspace/model.*"):
        download_model(name=name)


@pytest.mark.parametrize(
    ("model", "model_path", "verbose"),
    [
        ("path/to/checkpoint", "path/to/checkpoint", False),
        # (BoringModel(), "%s/BoringModel.ckpt"),
        (Module(), f"%s{os.path.sep}Module.pth", True),
        (svm.SVC(), f"%s{os.path.sep}SVC.pkl", 1),
    ],
)
@mock.patch("litmodels.io.cloud.sdk_upload_model")
def test_upload_model(mock_upload_model, tmp_path, model, model_path, verbose):
    mock_upload_model.return_value.name = "org-name/teamspace/model-name"

    # The lit-logger function is just a wrapper around the SDK function
    upload_model(
        model=model,
        name="org-name/teamspace/model-name",
        cloud_account="cluster_id",
        staging_dir=str(tmp_path),
        verbose=verbose,
    )
    expected_path = model_path % str(tmp_path) if "%" in model_path else model_path
    mock_upload_model.assert_called_once_with(
        path=expected_path,
        name="org-name/teamspace/model-name",
        cloud_account="cluster_id",
        progress_bar=True,
    )


@mock.patch("litmodels.io.cloud.sdk_download_model")
def test_download_model(mock_download_model):
    # The lit-logger function is just a wrapper around the SDK function
    download_model(
        name="org-name/teamspace/model-name",
        download_dir="where/to/download",
    )
    mock_download_model.assert_called_once_with(
        name="org-name/teamspace/model-name", download_dir="where/to/download", progress_bar=True
    )
