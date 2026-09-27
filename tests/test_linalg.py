import numpy as np
import pytest

from pid_tuneopt.identification.least_squares.linalg import _solve_normal_equations


def test_exact_solution():
    A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    b = np.array([1.0, 2.0, 3.0])
    x = _solve_normal_equations(A, b)
    np.testing.assert_allclose(x, [1.0, 2.0], atol=1e-12)


def test_overdetermined_noisy_data():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(100, 3))
    x_true = np.array([1.5, -0.5, 2.0])
    b = A @ x_true
    x = _solve_normal_equations(A, b)
    np.testing.assert_allclose(x, x_true, atol=1e-10)


def test_single_column():
    A = np.array([[1.0], [2.0], [3.0]])
    b = np.array([2.0, 4.0, 6.0])
    x = _solve_normal_equations(A, b)
    np.testing.assert_allclose(x, [2.0], atol=1e-12)


def test_singular_raises():
    A = np.array([[1.0, 2.0], [2.0, 4.0]])  # rank deficient
    b = np.array([1.0, 2.0])
    with pytest.raises(np.linalg.LinAlgError):
        _solve_normal_equations(A, b)