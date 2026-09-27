import numpy as np
import pytest

from pid_tuneopt.models.arx import arx_sim


# ---------------------------------------------------------------------------
# Independent brute-force reference implementation
# ---------------------------------------------------------------------------
def _reference(A, B, gamma_list, T, Xs, X_inits, Y_init):
    """Straight port of the original Cython logic using plain Python lists."""
    A = np.asarray(A, dtype=float).ravel()
    B = np.asarray(B, dtype=float)
    Xs = np.asarray(Xs, dtype=float)
    X_inits = np.atleast_2d(np.asarray(X_inits, dtype=float))

    ny = A.shape[0]
    nu = B.shape[1]
    ninputs = Xs.shape[1]
    mingamma = min(gamma_list)

    cols = []
    for i in range(ninputs):
        g = gamma_list[i]
        base = list(Xs[: len(Xs) - 1 - g, i])
        if g > mingamma:
            base = [X_inits[0, i]] * (g - mingamma) + base
        if nu > 1:
            base = [X_inits[0, i]] * (nu - 1) + base
        cols.append(base)

    n_rows = len(cols[0])
    Xmat = [[cols[i][r] for i in range(ninputs)] for r in range(n_rows)]

    n = len(T)
    y = [0.0] * n
    yb = [Y_init] * ny
    for j in range(mingamma + 1):
        y[j] = Y_init

    k = mingamma + 1
    m = nu
    while k < n:
        s = 0.0
        for j in range(ny):
            s += A[j] * yb[j]
        for i in range(ninputs):
            for j in range(nu):
                s += B[i, j] * Xmat[m - nu + j][i]
        y[k] = s
        for j in range(ny - 1):
            yb[j] = yb[j + 1]
        yb[ny - 1] = y[k]
        k += 1
        m += 1
    return np.array(y)


# ---------------------------------------------------------------------------
# Shape / dtype / initial conditions
# ---------------------------------------------------------------------------
class TestBasics:

    def test_output_shape_matches_T(self):
        A = np.array([0.5])
        B = np.array([[1.0]])
        T = np.arange(12, dtype=float)
        Xs = np.ones((12, 1))
        X_inits = np.zeros((1, 1))
        ypred = arx_sim(A, B, [0], T, Xs, X_inits, 0.0)
        assert ypred.shape == (12,)

    def test_output_dtype_is_float(self):
        ypred = arx_sim(
            np.array([0.5]), 
            np.array([[1.0]]), 
            [0], 
            np.arange(5, dtype=float), 
            np.zeros((5, 1)), 
            np.zeros((1, 1)), 
            0.0,
        )
        assert ypred.dtype == np.float64

    def test_first_element_is_Y_init_when_no_delay(self):
        ypred = arx_sim(
            np.array([0.5]), 
            np.array([[1.0]]), 
            [0], 
            np.arange(5, dtype=float), 
            np.zeros((5, 1)), 
            np.zeros((1, 1)),  
            42.0,
        )
        assert ypred[0] == 42.0

    def test_first_mingamma_plus_one_are_Y_init(self):
        gamma = 3
        ypred = arx_sim(
            np.array([0.5]), 
            np.array([[1.0]]), 
            [gamma], 
            np.arange(10, dtype=float),
            np.zeros((10, 1)), 
            np.zeros((1, 1)), 
            7.0,
        )
        np.testing.assert_array_equal(ypred[: gamma + 1], 7.0)

    def test_zero_coefficients_produces_zeros(self):
        ypred = arx_sim(
            np.array([0.0]), 
            np.array([[0.0]]), 
            [0], 
            np.arange(6, dtype=float),
            np.ones((6, 1)), 
            np.zeros((1, 1)), 
            5.0,
        )
        expected = np.zeros(6)
        expected[0] = 5.0
        np.testing.assert_allclose(ypred, expected)


# ---------------------------------------------------------------------------
# First-order ARX (ny = 1)
# ---------------------------------------------------------------------------
class TestFirstOrder:

    def test_recurrence_matches_closed_form(self):
        a, b = 0.5, 2.0
        A, B = np.array([a]), np.array([[b]])
        T = np.arange(6, dtype=float)
        u = np.arange(1, 7, dtype=float).reshape(-1, 1)
        ypred = arx_sim(A, B, [0], T, u, np.zeros((1, 1)), 1.0)

        expected = np.zeros(6)
        expected[0] = 1.0
        for k in range(1, 6):
            expected[k] = a * expected[k - 1] + b * u[k - 1, 0]
        np.testing.assert_allclose(ypred, expected)

    def test_exponential_decay_with_zero_input(self):
        a = 0.5
        ypred = arx_sim(
            np.array([a]), 
            np.array([[0.0]]), 
            [0], 
            np.arange(5, dtype=float),
            np.zeros((5, 1)), 
            np.zeros((1, 1)), 1.0,
        )
        expected = a ** np.arange(5)
        np.testing.assert_allclose(ypred, expected)


# ---------------------------------------------------------------------------
# Second-order ARX (ny = 2)
# ---------------------------------------------------------------------------
class TestSecondOrder:

    def test_fibonacci_sequence(self):
        # A = [1, 1], B = 0, Y_init = 1  ->  1, 2, 3, 5, 8, 13, ...
        ypred = arx_sim(
            np.array([1.0, 1.0]), 
            np.array([[0.0]]), 
            [0], 
            np.arange(6, dtype=float),
            np.zeros((6, 1)), 
            np.zeros((1, 1)), 
            1.0,
        )
        np.testing.assert_allclose(ypred, [1, 2, 3, 5, 8, 13])


# ---------------------------------------------------------------------------
# Delays (gamma_list)
# ---------------------------------------------------------------------------
class TestGamma:

    def test_warmup_region_uses_Y_init(self):
        ypred = arx_sim(
            np.array([0.5]), 
            np.array([[1.0]]), 
            [2], 
            np.arange(10, dtype=float),
            np.zeros((10, 1)), 
            np.zeros((1, 1)), 
            3.0,
        )
        np.testing.assert_array_equal(ypred[:3], [3.0, 3.0, 3.0])

    def test_delayed_input_starts_from_index_zero(self):
        a = 0.5
        Y_init = 5.0
        u = np.arange(10, dtype=float).reshape(-1, 1)
        ypred = arx_sim(
            np.array([a]), 
            np.array([[1.0]]), 
            [2], 
            np.arange(10, dtype=float),
            u, 
            np.zeros((1, 1)), 
            Y_init,
        )

        expected = np.full(10, 0.0)
        expected[:3] = Y_init
        for k in range(3, 10):
            expected[k] = a * expected[k - 1] + u[k - 3, 0]
        np.testing.assert_allclose(ypred, expected)


# ---------------------------------------------------------------------------
# Input lags (nu > 1)
# ---------------------------------------------------------------------------
class TestInputLags:

    def test_nu_two(self):
        a = 0.5
        b0, b1 = 1.0, 2.0
        n = 6
        u = np.arange(n, dtype=float).reshape(-1, 1)
        X_inits = np.array([[10.0]])  # used for the oldest lag at t = 0
        ypred = arx_sim(
            np.array([a]), 
            np.array([[b0, b1]]), 
            [0], 
            np.arange(n, dtype=float),
            u, 
            X_inits, 
            0.0,
        )

        expected = np.zeros(n)
        for k in range(1, n):
            u_km1 = u[k - 1, 0]
            u_km2 = X_inits[0, 0] if k == 1 else u[k - 2, 0]
            expected[k] = a * expected[k - 1] + b0 * u_km2 + b1 * u_km1
        np.testing.assert_allclose(ypred, expected)


# ---------------------------------------------------------------------------
# Multiple inputs
# ---------------------------------------------------------------------------
class TestMultiInput:

    def test_two_inputs_no_delay(self):
        a = 0.5
        b1, b2 = 1.0, 2.0
        n = 6
        Xs = np.column_stack([
            np.arange(1, n + 1, dtype=float),
            np.arange(11, 11 + n, dtype=float),
        ])
        # B has shape (ninputs, nu) = (2, 1)
        B = [[b1],
             [b2]]
        ypred = arx_sim(
            np.array([a]), 
            np.array(B), 
            [0, 0], 
            np.arange(n, dtype=float),
            Xs, 
            np.zeros((1, 2)), 
            0.0,
        )

        expected = np.zeros(n)
        for k in range(1, n):
            expected[k] = (a * expected[k - 1]
                           + b1 * Xs[k - 1, 0]
                           + b2 * Xs[k - 1, 1])
        np.testing.assert_allclose(ypred, expected)

    def test_two_inputs_different_gammas(self):
        a = 0.5
        b = 1.0
        n = 8
        Xs = np.column_stack([
            np.arange(1, n + 1, dtype=float),
            np.arange(101, 101 + n, dtype=float),
        ])
        X_inits = np.array([[0.0, 999.0]])
        # B has shape (ninputs, nu) = (2, 1)
        B = [[b],
             [b]]
        ypred = arx_sim(
            np.array([a]), 
            np.array(B), 
            [0, 2], 
            np.arange(n, dtype=float),
            Xs, 
            X_inits, 
            0.0,
        )

        expected = np.zeros(n)
        expected[1] = Xs[0, 0] + X_inits[0, 1]
        expected[2] = a * expected[1] + Xs[1, 0] + X_inits[0, 1]
        for k in range(3, n):
            expected[k] = a * expected[k - 1] + Xs[k - 1, 0] + Xs[k - 3, 1]
        np.testing.assert_allclose(ypred, expected)


# ---------------------------------------------------------------------------
# Comparison with the brute-force reference
# ---------------------------------------------------------------------------
class TestAgainstReference:

    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    def test_random_case(self, seed):
        rng = np.random.default_rng(seed)
        n = 30
        ninputs = 3
        ny = 2
        nu = 3
        A = rng.standard_normal(ny)
        B = rng.standard_normal((ninputs, nu))
        gamma_list = [0, 1, 2]
        T = np.arange(n, dtype=float)
        Xs = rng.standard_normal((n, ninputs))
        X_inits = rng.standard_normal((1, ninputs))
        Y_init = 0.5

        ypred = arx_sim(A, B, gamma_list, T, Xs, X_inits, Y_init)
        yref = _reference(A, B, gamma_list, T, Xs, X_inits, Y_init)
        np.testing.assert_allclose(ypred, yref, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__])
