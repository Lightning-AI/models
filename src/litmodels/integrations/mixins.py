import pickle
import tempfile
import warnings
from abc import ABC
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from litmodels import download_model, upload_model

if TYPE_CHECKING:
    import torch


class ModelRegistryMixin(ABC):
    """Mixin for model registry integration."""

    def push_to_registry(
        self, model_name: Optional[str] = None, model_version: Optional[str] = None, temp_folder: Optional[str] = None
    ) -> None:
        """Push the model to the registry.

        Args:
            model_name: The name of the model. If not use the class name.
            model_version: The version of the model. If None, the latest version is used.
            temp_folder: The temporary folder to save the model. If None, a default temporary folder is used.
        """

    @classmethod
    def pull_from_registry(
        cls, model_name: str, model_version: Optional[str] = None, temp_folder: Optional[str] = None
    ) -> object:
        """Pull the model from the registry.

        Args:
            model_name: The name of the model.
            model_version: The version of the model. If None, the latest version is used.
            temp_folder: The temporary folder to save the model. If None, a default temporary folder is used.
        """


class PickleRegistryMixin(ABC):
    """Mixin for pickle registry integration."""

    def push_to_registry(
        self, model_name: Optional[str] = None, model_version: Optional[str] = None, temp_folder: Optional[str] = None
    ) -> None:
        """Push the model to the registry.

        Args:
            model_name: The name of the model. If not use the class name.
            model_version: The version of the model. If None, the latest version is used.
            temp_folder: The temporary folder to save the model. If None, a default temporary folder is used.
        """
        if model_name is None:
            model_name = self.__class__.__name__
        if temp_folder is None:
            temp_folder = tempfile.gettempdir()
        pickle_path = Path(temp_folder) / f"{model_name}.pkl"
        with open(pickle_path, "wb") as fp:
            pickle.dump(self, fp, protocol=pickle.HIGHEST_PROTOCOL)
        model_registry = f"{model_name}:{model_version}" if model_version else model_name
        upload_model(name=model_registry, model=pickle_path)

    @classmethod
    def pull_from_registry(
        cls, model_name: str, model_version: Optional[str] = None, temp_folder: Optional[str] = None
    ) -> object:
        """Pull the model from the registry.

        Args:
            model_name: The name of the model.
            model_version: The version of the model. If None, the latest version is used.
            temp_folder: The temporary folder to save the model. If None, a default temporary folder is used.
        """
        if temp_folder is None:
            temp_folder = tempfile.gettempdir()
        model_registry = f"{model_name}:{model_version}" if model_version else model_name
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


class PyTorchRegistryMixin(ABC):
    """Mixin for PyTorch model registry integration."""

    def __post_init__(self) -> None:
        """Post-initialization method to set up the model."""
        import torch

        # Ensure that the model is in evaluation mode
        if not isinstance(self, torch.nn.Module):
            raise TypeError(f"The model must be a PyTorch `nn.Module` but got: {type(self)}")

    def push_to_registry(
        self, model_name: Optional[str] = None, model_version: Optional[str] = None, temp_folder: Optional[str] = None
    ) -> None:
        """Push the model to the registry.

        Args:
            model_name: The name of the model. If not use the class name.
            model_version: The version of the model. If None, the latest version is used.
            temp_folder: The temporary folder to save the model. If None, a default temporary folder is used.
        """
        import torch

        if not isinstance(self, torch.nn.Module):
            raise TypeError(f"The model must be a PyTorch `nn.Module` but got: {type(self)}")

        if model_name is None:
            model_name = self.__class__.__name__
        if temp_folder is None:
            temp_folder = tempfile.gettempdir()
        torch_path = Path(temp_folder) / f"{model_name}.pth"
        torch.save(self.state_dict(), torch_path)
        # todo: dump also object creation arguments so we can dump it and load with model for object instantiation
        model_registry = f"{model_name}:{model_version}" if model_version else model_name
        upload_model(name=model_registry, model=torch_path)

    @classmethod
    def pull_from_registry(
        cls,
        model_name: str,
        model_version: Optional[str] = None,
        temp_folder: Optional[str] = None,
        torch_load_kwargs: Optional[dict] = None,
    ) -> "torch.nn.Module":
        """Pull the model from the registry.

        Args:
            model_name: The name of the model.
            model_version: The version of the model. If None, the latest version is used.
            temp_folder: The temporary folder to save the model. If None, a default temporary folder is used.
            torch_load_kwargs: Additional arguments to pass to `torch.load()`.
        """
        import torch

        if temp_folder is None:
            temp_folder = tempfile.gettempdir()
        model_registry = f"{model_name}:{model_version}" if model_version else model_name
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
