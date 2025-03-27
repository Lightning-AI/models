import inspect
import json
import pickle
import tempfile
import warnings
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

from lightning_utilities.core.rank_zero import rank_zero_warn

from litmodels.io.cloud import download_model_files, upload_model_files

if TYPE_CHECKING:
    import torch


class ModelRegistryMixin(ABC):
    """Mixin for model registry integration."""

    def push_to_registry(
        self, name: Optional[str] = None, version: Optional[str] = None, temp_folder: Union[str, Path, None] = None
    ) -> None:
        """Push the model to the registry.

        Args:
            name: The name of the model. If not use the class name.
            version: The version of the model. If None, the latest version is used.
            temp_folder: The temporary folder to save the model. If None, a default temporary folder is used.
        """

    @classmethod
    def pull_from_registry(
        cls, name: str, version: Optional[str] = None, temp_folder: Union[str, Path, None] = None
    ) -> object:
        """Pull the model from the registry.

        Args:
            name: The name of the model.
            version: The version of the model. If None, the latest version is used.
            temp_folder: The temporary folder to save the model. If None, a default temporary folder is used.
        """

    def _setup(
        self, name: Optional[str] = None, temp_folder: Union[str, Path, None] = None
    ) -> Tuple[str, str, Union[str, Path]]:
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
        self, name: Optional[str] = None, version: Optional[str] = None, temp_folder: Union[str, Path, None] = None
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
        upload_model_files(name=name, path=pickle_path)

    @classmethod
    def pull_from_registry(
        cls, name: str, version: Optional[str] = None, temp_folder: Union[str, Path, None] = None
    ) -> object:
        """Pull the model from the registry.

        Args:
            name: The name of the model.
            version: The version of the model. If None, the latest version is used.
            temp_folder: The temporary folder to save the model. If None, a default temporary folder is used.
        """
        if temp_folder is None:
            temp_folder = tempfile.mkdtemp()
        model_registry = f"{name}:{version}" if version else name
        files = download_model_files(name=model_registry, download_dir=temp_folder)
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

    def __new__(cls, *args: Any, **kwargs: Any) -> "torch.nn.Module":
        """Create a new instance of the class without calling __init__."""
        instance = super().__new__(cls)

        # Get __init__ signature excluding 'self'
        init_sig = inspect.signature(cls.__init__)
        params = list(init_sig.parameters.values())[1:]  # Skip self

        # Create temporary signature for binding
        temp_sig = init_sig.replace(parameters=params)

        # Bind and apply defaults
        bound_args = temp_sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Store unified kwargs
        instance.__init_kwargs = bound_args.arguments
        return instance

    def push_to_registry(
        self, name: Optional[str] = None, version: Optional[str] = None, temp_folder: Union[str, Path, None] = None
    ) -> None:
        """Push the model to the registry.

        Args:
            name: The name of the model. If not use the class name.
            version: The version of the model. If None, the latest version is used.
            temp_folder: The temporary folder to save the model. If None, a default temporary folder is used.
        """
        import torch

        # Ensure that the model is in evaluation mode
        if not isinstance(self, torch.nn.Module):
            raise TypeError(f"The model must be a PyTorch `nn.Module` but got: {type(self)}")

        name, model_name, temp_folder = self._setup(name, temp_folder)

        if self.__init_kwargs:
            try:
                # Save the model arguments to a JSON file
                init_kwargs_path = Path(temp_folder) / f"{model_name}__init_kwargs.json"
                with open(init_kwargs_path, "w") as fp:
                    json.dump(self.__init_kwargs, fp)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to save model arguments: {e}."
                    " Ensure the model's arguments are JSON serializable or use `PickleRegistryMixin`."
                ) from e
        elif not hasattr(self, "__init_kwargs"):
            rank_zero_warn(
                "The child class is missing `__init_kwargs`."
                " Ensure `PyTorchRegistryMixin` is first in the inheritance order"
                " or call `PyTorchRegistryMixin.__init__` explicitly in the child class."
            )

        torch_state_dict_path = Path(temp_folder) / f"{model_name}.pth"
        torch.save(self.state_dict(), torch_state_dict_path)
        model_registry = f"{name}:{version}" if version else name
        # todo: consider creating another temp folder and copying these two files
        # todo: updating SDK to support uploading just specific files
        upload_model_files(name=model_registry, path=temp_folder)

    @classmethod
    def pull_from_registry(
        cls,
        name: str,
        version: Optional[str] = None,
        temp_folder: Union[str, Path, None] = None,
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
        files = download_model_files(name=model_registry, download_dir=temp_folder)

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

        init_files = [fp for fp in files if fp.endswith("__init_kwargs.json")]
        if not init_files:
            init_kwargs = {}
        elif len(init_files) > 1:
            raise RuntimeError(f"Multiple init files found for model: {model_registry} with {init_files}")
        else:
            init_kwargs_path = Path(temp_folder) / init_files[0]
            with open(init_kwargs_path) as fp:
                init_kwargs = json.load(fp)

        # Create a new model instance without calling __init__
        instance = cls(**init_kwargs)
        if not isinstance(instance, torch.nn.Module):
            raise TypeError(f"The model must be a PyTorch `nn.Module` but got: {type(instance)}")
        # Now load the state dict on the instance
        instance.load_state_dict(state_dict, strict=True)
        return instance
