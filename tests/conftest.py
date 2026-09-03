"""Shared pytest configuration.

Tests that train a network are marked ``slow``.  They still run by default -- a reference
implementation whose training path is untested is not much of a reference -- but they can
be skipped with ``pytest -m "not slow"`` for a fast inner loop.
"""

from __future__ import annotations

import pytest
import torch


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: exercises a training loop")


@pytest.fixture(autouse=True)
def _deterministic_and_single_threaded() -> None:
    """Keep tests reproducible and avoid thread oversubscription on CI runners."""
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(False)
    torch.set_num_threads(2)