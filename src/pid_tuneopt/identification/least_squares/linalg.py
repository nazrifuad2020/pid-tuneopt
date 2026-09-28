from __future__ import annotations

import numpy as np


def _solve_normal_equations(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Least-squares solution of ``A x = b`` via the normal equations."""
    return np.linalg.inv(A.T @ A) @ A.T @ b