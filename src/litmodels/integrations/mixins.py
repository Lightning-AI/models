import pickle
import tempfile
from abc import ABC
from pathlib import Path
from typing import Optional, Any

from litmodels import download_model, upload_model


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
