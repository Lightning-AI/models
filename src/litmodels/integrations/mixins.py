import pickle
import tempfile
import warnings
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

from litmodels import download_model, upload_model

if TYPE_CHECKING:
    import torch


class ModelRegistryMixin(ABC):
    """Mixin for model registry integration."""

    def push_to_registry(
        self, name: Optional[str] = None, version: Optional[str] = None, temp_folder: Optional[str] = None
    ) -> None:
        """Push the model to the registry.

        Args:
            name: The name of the model. If not use the class name.
            version: The version of the model. If None, the latest version is used.
            temp_folder: The temporary folder to save the model. If None, a default temporary folder is used.
        """

    @classmethod
    def pull_from_registry(cls, name: str, version: Optional[str] = None, temp_folder: Optional[str] = None) -> object:
        """Pull the model from the registry.

        Args:
            name: The name of the model.
            version: The version of the model. If None, the latest version is used.
            temp_folder: The temporary folder to save the model. If None, a default temporary folder is used.
        """

    def _setup(self, name: Optional[str] = None, temp_folder: Optional[str] = None) -> Tuple[str, str, str]:
        """Parse and validate the model name and temporary folder."""
        if name is None:
            name = model_name = self.__class__.__name__
        elif ":" in name:
            raise ValueError(f"Invalid model name: '{name}'. It should not contain ':' associated with version.")
        else:
            model_name = name.split("/")[-1]
        if temp_folder is None:
            temp_folder = tempfile.mkdtemp()
        return name, model_name, temp_folder


class PickleRegistryMixin(ModelRegistryMixin):
    """Mixin for pickle registry integration."""

    def push_to_registry(
        self, name: Optional[str] = None, version: Optional[str] = None, temp_folder: Optional[str] = None
    ) -> None:
        """Push the model to the registry.

        Args:
            name: The name of the model. If not use the class name.
            version: The version of the model. If None, the latest version is used.
            temp_folder: The temporary folder to save the model. If None, a default temporary folder is used.
        """
        name, model_name, temp_folder = self._setup(name, temp_folder)
        pickle_path = Path(temp_folder) / f"{model_name}.pkl"
        with open(pickle_path, "wb") as fp:
            pickle.dump(self, fp, protocol=pickle.HIGHEST_PROTOCOL)
        if version:
            name = f"{name}:{version}"
        upload_model(name=name, model=pickle_path)

    @classmethod
    def pull_from_registry(cls, name: str, version: Optional[str] = None, temp_folder: Optional[str] = None) -> object:
        """Pull the model from the registry.

        Args:
            name: The name of the model.
            version: The version of the model. If None, the latest version is used.
            temp_folder: The temporary folder to save the model. If None, a default temporary folder is used.
        """
        if temp_folder is None:
            temp_folder = tempfile.mkdtemp()
        model_registry = f"{name}:{version}" if version else name
        files = download_model(name=model_registry, download_dir=temp_folder)
        pkl_files = [f for f in files if f.endswith(".pkl")]
        if not pkl_files:
            raise RuntimeError(f"No pickle file found for model: {model_registry} with {files}")
        if len(pkl_files) > 1:
            raise RuntimeError(f"Multiple pickle files found for model: {model_registry} with {pkl_files}")
        pkl_path = Path(temp_folder) / pkl_files[0]
        with open(pkl_path, "rb") as fp:
            obj = pickle.load(fp)
        if not isinstance(obj, cls):
            raise RuntimeError(f"Unpickled object is not of type {cls.__name__}: {type(obj)}")
        return obj


class PyTorchRegistryMixin(ModelRegistryMixin):
    """Mixin for PyTorch model registry integration."""

    def push_to_registry(
        self, name: Optional[str] = None, version: Optional[str] = None, temp_folder: Optional[str] = None
    ) -> None:
        """Push the model to the registry.

        Args:
            name: The name of the model. If not use the class name.
            version: The version of the model. If None, the latest version is used.
            temp_folder: The temporary folder to save the model. If None, a default temporary folder is used.
        """
        import torch

        if not isinstance(self, torch.nn.Module):
            raise TypeError(f"The model must be a PyTorch `nn.Module` but got: {type(self)}")

        name, model_name, temp_folder = self._setup(name, temp_folder)
        torch_path = Path(temp_folder) / f"{model_name}.pth"
        torch.save(self.state_dict(), torch_path)
        # todo: dump also object creation arguments so we can dump it and load with model for object instantiation
        model_registry = f"{name}:{version}" if version else name
        upload_model(name=model_registry, model=torch_path)

    @classmethod
    def pull_from_registry(
        cls,
        name: str,
        version: Optional[str] = None,
        temp_folder: Optional[str] = None,
        torch_load_kwargs: Optional[dict] = None,
    ) -> "torch.nn.Module":
        """Pull the model from the registry.

        Args:
            name: The name of the model.
            version: The version of the model. If None, the latest version is used.
            temp_folder: The temporary folder to save the model. If None, a default temporary folder is used.
            torch_load_kwargs: Additional arguments to pass to `torch.load()`.
        """
        import torch

        if temp_folder is None:
            temp_folder = tempfile.mkdtemp()
        model_registry = f"{name}:{version}" if version else name
        files = download_model(name=model_registry, download_dir=temp_folder)
        torch_files = [f for f in files if f.endswith(".pth")]
        if not torch_files:
            raise RuntimeError(f"No torch file found for model: {model_registry} with {files}")
        if len(torch_files) > 1:
            raise RuntimeError(f"Multiple torch files found for model: {model_registry} with {torch_files}")
        state_dict_path = Path(temp_folder) / torch_files[0]
        # ignore future warning about changed default
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            state_dict = torch.load(state_dict_path, **(torch_load_kwargs if torch_load_kwargs else {}))

        # Create a new model instance without calling __init__
        instance = cls()  # todo: we need to add args used when created dumped model
        if not isinstance(instance, torch.nn.Module):
            raise TypeError(f"The model must be a PyTorch `nn.Module` but got: {type(instance)}")
        # Now load the state dict on the instance
        instance.load_state_dict(state_dict, strict=True)
        return instance
