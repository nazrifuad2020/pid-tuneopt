import numpy as np
import pytest

from pid_tuneopt.identification.least_squares.fits import (
    linear_least_squares_fit_first_order,
    linear_least_squares_fit_second_order,
    linear_least_squares_fit_integrating,
)


def _gen_first_order(a1, b1, u):
    """y[k] = a1*y[k-1] + b1*u[k-2] with zero initial conditions."""
    n = len(u)
    y = np.zeros(n)
    for k in range(2, n):
        y[k] = a1 * y[k - 1] + b1 * u[k - 2]
    return y


def _gen_second_order(a1, a2, b1, b2, u):
    """y[k] = a1*y[k-1] + a2*y[k-2] + b1*u[k-2] + b2*u[k-3]."""
    n = len(u)
    y = np.zeros(n)
    for k in range(2, n):
        u1 = u[k - 2] if k - 2 >= 0 else 0.0
        u2 = u[k - 3] if k - 3 >= 0 else 0.0
        y[k] = a1 * y[k - 1] + a2 * y[k - 2] + b1 * u1 + b2 * u2
    return y


class TestFirstOrder:
    def test_recovers_known_parameters(self):
        rng = np.random.default_rng(0)
        u = np.concatenate(([0.0], rng.normal(size=200)))
        a1_true, b1_true = 0.7, 0.4
        y = _gen_first_order(a1_true, b1_true, u)

        Xdot = u.reshape(-1, 1)
        Ydot = y
        Xinit = np.zeros((1, 1))

        a1, b1 = linear_least_squares_fit_first_order(
            Xdot, Ydot, [1], Xinit
        )
        assert np.isclose(a1, a1_true, atol=1e-10)
        assert np.isclose(b1[0], b1_true, atol=1e-10)

    def test_returns_scalar_and_list(self):
        u = np.arange(20, dtype=float)
        y = _gen_first_order(0.5, 0.3, u)
        a1, b1 = linear_least_squares_fit_first_order(
            u.reshape(-1, 1), y, [1], np.zeros((1, 1))
        )
        assert np.isscalar(a1) or np.ndim(a1) == 0
        assert isinstance(b1, list)
        assert len(b1) == 1

    def test_two_inputs(self):
        rng = np.random.default_rng(1)
        u = np.column_stack([
            np.concatenate(([0.0], rng.normal(size=200))),
            np.concatenate(([0.0], rng.normal(size=200))),
        ])
        a1, b1_true, b2_true = 0.6, 0.3, 0.5
        # y[k] = a1*y[k-1] + b1*u0[k-2] + b2*u1[k-2]
        n = len(u)
        y = np.zeros(n)
        for k in range(2, n):
            y[k] = a1 * y[k - 1] + b1_true * u[k - 2, 0] + b2_true * u[k - 2, 1]

        a1_fit, b_fit = linear_least_squares_fit_first_order(
            u, y, [1, 1], np.zeros((1, 2))
        )
        assert np.isclose(a1_fit, a1, atol=1e-10)
        assert np.isclose(b_fit[0], b1_true, atol=1e-10)
        assert np.isclose(b_fit[1], b2_true, atol=1e-10)


class TestSecondOrder:
    def test_recovers_known_parameters(self):
        rng = np.random.default_rng(2)
        u = np.concatenate(([0.0], rng.normal(size=300)))
        a1_t, a2_t, b1_t, b2_t = 0.5, 0.2, 0.3, 0.1
        y = _gen_second_order(a1_t, a2_t, b1_t, b2_t, u)

        a1, a2, b1, b2 = linear_least_squares_fit_second_order(
            u.reshape(-1, 1), y, [1], np.zeros((1, 1))
        )
        assert np.isclose(a1, a1_t, atol=1e-10)
        assert np.isclose(a2, a2_t, atol=1e-10)
        assert np.isclose(b1[0], b1_t, atol=1e-10)
        assert np.isclose(b2[0], b2_t, atol=1e-10)

    def test_returns_lists_for_each_input(self):
        u = np.zeros((20, 2))
        u[:, 0] = np.arange(20)
        u[:, 1] = np.linspace(0, 1, 20)
        y = _gen_second_order(0.5, 0.1, 0.3, 0.05, u[:, 0])
        a1, a2, b1, b2 = linear_least_squares_fit_second_order(
            u, y, [1, 1], np.zeros((1, 2))
        )
        assert isinstance(b1, list) and isinstance(b2, list)
        assert len(b1) == 2 and len(b2) == 2


class TestIntegrating:
    def test_recovers_pure_gain(self):
        rng = np.random.default_rng(3)
        u = rng.normal(size=300).reshape(-1, 1)
        gain = 0.42
        # Fit enforces: Δy[m] = b · u[m-1]   (gamma = mingamma = 0)
        y = np.concatenate(([0.0], np.cumsum(gain * u[:-1, 0])))

        _, gains = linear_least_squares_fit_integrating(
            u, y, [0], np.zeros((1, 1))
        )
        assert np.isclose(gains[0], gain, atol=1e-10)

    def test_returns_a1_equal_one(self):
        u = np.arange(50, dtype=float).reshape(-1, 1)
        y = np.concatenate(([0.0], np.cumsum(u[:-1, 0])))
        a1, _ = linear_least_squares_fit_integrating(
            u, y, [0], np.zeros((1, 1))
        )
        assert a1 == 1.0

    def test_two_inputs_recover_both_gains(self):
        rng = np.random.default_rng(4)
        u = rng.normal(size=(200, 2))
        g = np.array([0.5, -0.25])
        # Δy[m] = g · u[m-1]
        y = np.concatenate(([0.0], np.cumsum(u[:-1] @ g)))

        _, gains = linear_least_squares_fit_integrating(
            u, y, [0, 0], np.zeros((1, 2))
        )
        np.testing.assert_allclose(gains, g, atol=1e-10)

    def test_gamma_one_shifts_input_by_one(self):
        """gamma = mingamma = 1 ⇒ fit enforces Δy[m] = b · u[m-2]."""
        rng = np.random.default_rng(11)
        u = rng.normal(size=300).reshape(-1, 1)
        gain = 0.3

        # Build y so that y[m] - y[m-1] = gain * u[m-2] for m >= 2.
        # Equivalently: y[m] = gain * sum_{i=0}^{m-2} u[i].
        y = np.concatenate(([0.0, 0.0], np.cumsum(gain * u[:-2, 0])))

        _, gains = linear_least_squares_fit_integrating(
            u, y, [1], np.zeros((1, 1))
        )
        assert np.isclose(gains[0], gain, atol=1e-10)

    def test_gamma_alignment_rejects_wrong_delay(self):
        """A gamma=0 signal fitted with gamma=1 must NOT recover the gain.

        For white-noise u the mismatch turns the normal equations into a
        near-orthogonality, so the returned coefficient collapses to ~0.
        """
        rng = np.random.default_rng(12)
        u = rng.normal(size=300).reshape(-1, 1)
        gain = 0.3
        # gamma=0 data: Δy[m] = gain · u[m-1]
        y = np.concatenate(([0.0], np.cumsum(gain * u[:-1, 0])))

        _, gains = linear_least_squares_fit_integrating(
            u, y, [1], np.zeros((1, 1))
        )
        # Sanity: the coefficient is tiny, not the true gain.
        assert abs(gains[0]) < 0.05
        assert not np.isclose(gains[0], gain, atol=1e-3)