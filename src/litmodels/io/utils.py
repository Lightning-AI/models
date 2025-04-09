from pathlib import Path
from typing import Any, Union

from lightning_utilities import module_available
import pickle

if module_available("joblib"):
    import joblib
else:
    joblib = None



def dump_pickle(model: Any, path: Union[str, Path]) -> None:
    """Dump a model to a pickle file.

    Args:
        model: The model to be pickled.
        path: The path where the model will be saved.
    """
    if joblib is not None:
        joblib.dump(model, filename=path, compress=7)
    else:
        with open(path, "wb") as fp:
            pickle.dump(model, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Union[str, Path]) -> Any:
    """Load a model from a pickle file.

    Args:
        path: The path to the pickle file.

    Returns:
        The unpickled model.
    """
    if joblib is not None:
        return joblib.load(path)
    else:
        with open(path, "rb") as fp:
            return pickle.load(fp)