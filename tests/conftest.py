"""Shared pytest fixtures.

The dataset is loaded once per test session. If it has not been downloaded
yet, the data-dependent tests are skipped with a message saying how to get it
(tests should not silently start a 150 MB download).
"""

import pytest

from btc_aml.data import find_raw_dir, load_bundle
from btc_aml.utils import data_dir, load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config("configs/quick.yaml")


@pytest.fixture(scope="session")
def bundle(cfg):
    base = data_dir(cfg)
    cache = base / "processed" / "elliptic_bundle.pt"
    if not cache.exists() and find_raw_dir(base) is None:
        pytest.skip("Dataset not downloaded. Run: python -m btc_aml.data --config configs/quick.yaml")
    return load_bundle(cfg)
