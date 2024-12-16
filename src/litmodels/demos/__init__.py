"""Define a demos model for examples and testing purposes."""

from lightning_utilities import module_available

__all__ = []

if module_available("lightning"):
    from lightning.pytorch.demos.boring_classes import BoringModel, DemoModel

    __all__ += ["BoringModel", "DemoModel"]
elif module_available("pytorch_lightning"):
    from pytorch_lightning.demos.boring_classes import BoringModel, DemoModel

    __all__ += ["BoringModel", "DemoModel"]
