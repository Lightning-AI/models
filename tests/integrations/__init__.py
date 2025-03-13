import pytest
from litmodels.integrations.imports import _LIGHTNING_AVAILABLE, _PYTORCHLIGHTNING_AVAILABLE

_SKIP_IF_LIGHTNING_MISSING = pytest.mark.skipif(not _LIGHTNING_AVAILABLE, reason="Lightning not available")
_SKIP_IF_PYTORCHLIGHTNING_MISSING = pytest.mark.skipif(
    not _PYTORCHLIGHTNING_AVAILABLE, reason="PyTorch Lightning not available"
)
