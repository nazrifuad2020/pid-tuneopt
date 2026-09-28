"""Shared pytest fixtures for the least_squares test suite."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def single_input() -> np.ndarray:
    """One input over 10 samples (column vector)."""
    return np.arange(10, dtype=float).reshape(-1, 1)


@pytest.fixture
def two_inputs() -> np.ndarray:
    """Two independent inputs over 10 samples."""
    x1 = np.arange(10, dtype=float)
    x2 = 0.5 * np.arange(10, dtype=float) + 1.0
    return np.column_stack([x1, x2])


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic random generator."""
    return np.random.default_rng(seed=12345)