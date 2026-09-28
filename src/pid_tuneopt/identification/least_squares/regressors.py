from __future__ import annotations

import numpy as np

from .utils import _prepend_constant


def _delayed_input_column(
    X: np.ndarray,
    gamma: int,
    mingamma: int,
    init_value: float,
    col_idx: int,
) -> np.ndarray:
    """Return column ``col_idx`` of ``X`` delayed by ``gamma`` samples.

    If ``gamma > mingamma`` the column is left-padded with ``init_value`` so
    that every delayed column shares the same length.
    """
    col = X[: X.shape[0] - 1 - gamma, col_idx]
    if gamma > mingamma:
        col = _prepend_constant(col, gamma - mingamma, init_value)
    return col


def _build_input_regressor_matrix(
    X: np.ndarray,
    gamma_list: list[int],
    mingamma: int,
    init_row: np.ndarray,
) -> np.ndarray:
    """Stack one delayed column per input into a 2-D regressor matrix."""
    return np.column_stack([
        _delayed_input_column(X, int(gamma_list[i]), mingamma, init_row[0, i], i)
        for i in range(X.shape[1])
    ])