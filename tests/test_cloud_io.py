import os
from unittest import mock

import pytest
from litmodels.cloud_io import download_model, upload_model, upload_model_files
from torch.nn import Module


@pytest.mark.parametrize("name", ["org/model", "model-name", "/too/many/slashes"])
def test_wrong_model_name(name):
    with pytest.raises(ValueError, match=r".*organization/teamspace/model.*"):
        upload_model_files(path="path/to/checkpoint", name=name)
    with pytest.raises(ValueError, match=r".*organization/teamspace/model.*"):
        download_model(name=name)


@pytest.mark.parametrize(
    ("model", "model_path"),
    [
        ("path/to/checkpoint", "path/to/checkpoint"),
        # (BoringModel(), "%s/BoringModel.ckpt"),
        (Module(), f"%s{os.path.sep}Module.pth"),
    ],
)
def test_upload_model(mocker, tmpdir, model, model_path):
    # mocking the _get_teamspace to return another mock
    ts_mock = mock.MagicMock()
    mocker.patch("litmodels.cloud_io._get_teamspace", return_value=ts_mock)

    # The lit-logger function is just a wrapper around the SDK function
    upload_model(
        model,
        name="org-name/teamspace/model-name",
        cluster_id="cluster_id",
        staging_dir=tmpdir,
    )
    expected_path = model_path % str(tmpdir) if "%" in model_path else model_path
    ts_mock.upload_model.assert_called_once_with(
        path=expected_path,
        name="model-name",
        cluster_id="cluster_id",
        progress_bar=True,
    )


def test_download_model(mocker):
    # mocking the _get_teamspace to return another mock
    ts_mock = mock.MagicMock()
    mocker.patch("litmodels.cloud_io._get_teamspace", return_value=ts_mock)
    # The lit-logger function is just a wrapper around the SDK function
    download_model(
        name="org-name/teamspace/model-name",
        download_dir="where/to/download",
    )
    ts_mock.download_model.assert_called_once_with(
        name="model-name", download_dir="where/to/download", progress_bar=True
    )
