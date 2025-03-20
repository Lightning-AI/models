"""Integrations with training frameworks like PyTorch Lightning, TensorFlow, and others."""

from litmodels.integrations.imports import _LIGHTNING_AVAILABLE, _PYTORCHLIGHTNING_AVAILABLE

__all__ = []

if _LIGHTNING_AVAILABLE:
    from litmodels.integrations.checkpoints import LightningModelCheckpoint

    __all__ += ["LightningModelCheckpoint"]

if _PYTORCHLIGHTNING_AVAILABLE:
    from litmodels.integrations.checkpoints import PytorchLightningModelCheckpoint

    __all__ += ["PytorchLightningModelCheckpoint"]
