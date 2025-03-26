from unittest import mock

import torch
from litmodels.integrations.mixins import PickleRegistryMixin, PyTorchRegistryMixin
from torch import nn


class DummyModel(PickleRegistryMixin):
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, DummyModel) and self.value == other.value


@mock.patch("litmodels.integrations.mixins.upload_model")
@mock.patch("litmodels.integrations.mixins.download_model")
def test_pickle_push_and_pull(mock_download_model, mock_upload_model, tmp_path):
    # Create an instance of DummyModel and call push_to_registry.
    dummy = DummyModel(42)
    dummy.push_to_registry(version="v1", temp_folder=str(tmp_path))
    # The expected registry name is "dummy_model:v1" and the file should be placed in the temp folder.
    expected_path = tmp_path / "DummyModel.pkl"
    mock_upload_model.assert_called_once_with(name="DummyModel:v1", model=expected_path)

    # Set the mock to return the full path to the pickle file.
    mock_download_model.return_value = ["DummyModel.pkl"]
    # Call pull_from_registry and load the DummyModel instance.
    loaded_dummy = DummyModel.pull_from_registry(name="dummy_model", version="v1", temp_folder=str(tmp_path))
    # Verify that the unpickled instance has the expected value.
    assert loaded_dummy.value == 42


class DummyTorchModel(nn.Module, PyTorchRegistryMixin):
    def __init__(self, input_size=784):
        super().__init__()
        self.fc = nn.Linear(input_size, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


@mock.patch("litmodels.integrations.mixins.upload_model")
@mock.patch("litmodels.integrations.mixins.download_model")
def test_pytorch_pull_updated(mock_download_model, mock_upload_model, tmp_path):
    # Create an instance, push the model and record its forward output.
    dummy = DummyTorchModel(784)
    dummy.eval()
    input_tensor = torch.randn(1, 784)
    output_before = dummy(input_tensor)

    dummy.push_to_registry(temp_folder=str(tmp_path))
    expected_path = tmp_path / f"{dummy.__class__.__name__}.pth"
    mock_upload_model.assert_called_once_with(name="DummyTorchModel", model=expected_path)

    torch.save(dummy.state_dict(), expected_path)
    # Prepare mocking for pull_from_registry.
    mock_download_model.return_value = [f"{dummy.__class__.__name__}.pth"]
    loaded_dummy = DummyTorchModel.pull_from_registry(name="DummyTorchModel", temp_folder=str(tmp_path))
    loaded_dummy.eval()
    output_after = loaded_dummy(input_tensor)

    assert isinstance(loaded_dummy, DummyTorchModel)
    # Compare the outputs as a verification.
    assert torch.allclose(output_before, output_after), "Loaded model output differs from original."
