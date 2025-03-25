from unittest import mock

from litmodels.integrations.mixins import PickleRegistryMixin


class DummyModel(PickleRegistryMixin):
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, DummyModel) and self.value == other.value


@mock.patch("litmodels.integrations.mixins.upload_model")
@mock.patch("litmodels.integrations.mixins.download_model")
def test_mixin_push_and_pull(mock_download_model, mock_upload_model, tmp_path):
    # Create an instance of DummyModel and call push_to_registry.
    dummy = DummyModel(42)
    dummy.push_to_registry(model_version="v1", temp_folder=str(tmp_path))
    # The expected registry name is "dummy_model:v1" and the file should be placed in the temp folder.
    expected_path = tmp_path / "DummyModel.pkl"
    mock_upload_model.assert_called_once_with(name="DummyModel:v1", model=expected_path)

    # Set the mock to return the full path to the pickle file.
    mock_download_model.return_value = ["DummyModel.pkl"]
    # Call pull_from_registry and load the DummyModel instance.
    loaded_dummy = DummyModel.pull_from_registry(
        model_name="dummy_model", model_version="v1", temp_folder=str(tmp_path)
    )
    # Verify that the unpickled instance has the expected value.
    assert isinstance(loaded_dummy, DummyModel)
    assert loaded_dummy.value == 42
