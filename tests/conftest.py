"""Test isolation: no test may see a model API key."""
import os

import pytest


def _strip_keys() -> None:
    """Delete MODEL_API_KEY and every META_* env var."""
    for k in [k for k in os.environ if k == "MODEL_API_KEY" or k.startswith("META_")]:
        del os.environ[k]


@pytest.fixture(autouse=True)
def _no_api_key():
    """Strip keys before and after every test; assert none is visible."""
    _strip_keys()
    assert "MODEL_API_KEY" not in os.environ
    assert not [k for k in os.environ if k.startswith("META_")]
    yield
    _strip_keys()
