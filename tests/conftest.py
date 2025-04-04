"""Pytest configuration for integration tests."""

import pytest
from litmodels.integrations.checkpoints import get_model_manager


@pytest.fixture(autouse=True)
def reset_model_manager():
    get_model_manager.cache_clear()
    # Optionally, call it once to initialize immediately
    return get_model_manager()
