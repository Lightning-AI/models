from unittest import mock

import pytest
import torch
from torch import nn

from litmodels.integrations.mixins import PickleRegistryMixin, PyTorchRegistryMixin


class DummyModel(PickleRegistryMixin):
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, DummyModel) and self.value == other.value


@mock.patch("litmodels.integrations.mixins.upload_model_files")
@mock.patch("litmodels.integrations.mixins.download_model_files")
def test_pickle_push_and_pull(mock_download_model, mock_upload_model, tmp_path):
    # Create an instance of DummyModel and call push_to_registry.
    dummy = DummyModel(42)
    dummy.upload_model(version="v1", temp_folder=str(tmp_path))
    # The expected registry name is "dummy_model:v1" and the file should be placed in the temp folder.
    expected_path = tmp_path / "DummyModel.pkl"
    mock_upload_model.assert_called_once_with(name="DummyModel:v1", path=expected_path)

    # Set the mock to return the full path to the pickle file.
    mock_download_model.return_value = ["DummyModel.pkl"]
    # Call pull_from_registry and load the DummyModel instance.
    loaded_dummy = DummyModel.download_model(name="dummy_model", version="v1", temp_folder=str(tmp_path))
    # Verify that the unpickled instance has the expected value.
    assert loaded_dummy.value == 42


class DummyTorchModelFirst(PyTorchRegistryMixin, nn.Module):
    def __init__(self, input_size: int, output_size: int = 10):
        # PyTorchRegistryMixin.__init__ will capture these arguments
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


class DummyTorchModelSecond(nn.Module, PyTorchRegistryMixin):
    def __init__(self, input_size: int, output_size: int = 10):
        PyTorchRegistryMixin.__init__(input_size, output_size)
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


@pytest.mark.parametrize("torch_class", [DummyTorchModelFirst, DummyTorchModelSecond])
@mock.patch("litmodels.integrations.mixins.upload_model_files")
@mock.patch("litmodels.integrations.mixins.download_model_files")
def test_pytorch_push_and_pull(mock_download_model, mock_upload_model, torch_class, tmp_path):
    # Create an instance, push the model and record its forward output.
    dummy = torch_class(784)
    dummy.eval()
    input_tensor = torch.randn(1, 784)
    output_before = dummy(input_tensor)

    torch_file = f"{dummy.__class__.__name__}.pth"
    torch.save(dummy.state_dict(), tmp_path / torch_file)
    json_file = f"{dummy.__class__.__name__}__init_kwargs.json"
    json_path = tmp_path / json_file
    with open(json_path, "w") as fp:
        fp.write('{"input_size": 784, "output_size": 10}')

    dummy.upload_model(temp_folder=str(tmp_path))
    mock_upload_model.assert_called_once_with(
        name=torch_class.__name__, path=[tmp_path / f"{torch_class.__name__}.pth", json_path]
    )

    # Prepare mocking for pull_from_registry.
    mock_download_model.return_value = [torch_file, json_file]
    loaded_dummy = torch_class.download_model(name=torch_class.__name__, temp_folder=str(tmp_path))
    loaded_dummy.eval()
    output_after = loaded_dummy(input_tensor)

    assert isinstance(loaded_dummy, torch_class)
    # Compare the outputs as a verification.
    assert torch.allclose(output_before, output_after), "Loaded model output differs from original."
