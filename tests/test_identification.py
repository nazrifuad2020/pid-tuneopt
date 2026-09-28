"""Integration tests for ``sysid_MISO``.

Contract tests stub out ``arx_sim`` and inspect the call site; recovery
tests run the real ``arx_sim`` and verify the fitted coefficients and
predictions against the synthetic model that generated the data.
"""
from __future__ import annotations

import numpy as np
import pytest

from pid_tuneopt.identification.least_squares import (
    sysid_MISO,
    PREP_RAW,
    PREP_REMOVE_MEAN,
    PREP_SUBTRACT_INITIAL,
)
from pid_tuneopt.identification.least_squares import identification as ident_mod


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def captured_arx(monkeypatch):
    """Replace ``arx_sim`` with a recorder exposing the captured call."""

    class Captured:
        args = None
        kwargs = None
        calls = 0

    def _stub(*args, **kwargs):
        Captured.args = args
        Captured.kwargs = kwargs
        Captured.calls += 1
        T = args[3]
        return np.zeros(len(T))

    monkeypatch.setattr(ident_mod, "arx_sim", _stub)
    return Captured


# ---------------------------------------------------------------------
# Synthetic data generators (always return 2-D U)
# ---------------------------------------------------------------------

def _make_arx1_data(n=300, gamma=1, a1=0.7, b1=0.4, seed=0):
    """y[k] = a1*y[k-1] + b1*u[k-1-gamma], y[0]=0, u[:gamma+1]=0."""
    rng = np.random.default_rng(seed)
    T = np.arange(n, dtype=float)
    U = np.zeros((n, 1))                       # 2-D on purpose
    U[gamma + 1:, 0] = rng.normal(size=n - gamma - 1)
    Y = np.zeros(n)
    for k in range(gamma + 1, n):
        Y[k] = a1 * Y[k - 1] + b1 * U[k - 1 - gamma, 0]
    return T, U, Y


def _make_arx2_data(n=400, gamma=1, a1=0.7, a2=0.1,
                    b1=0.4, b2=0.05, seed=1):
    """y[k] = a1*y[k-1] + a2*y[k-2] + b1*u[k-1-gamma] + b2*u[k-2-gamma]."""
    rng = np.random.default_rng(seed)
    T = np.arange(n, dtype=float)
    U = np.zeros((n, 1))
    U[gamma + 2:, 0] = rng.normal(size=n - gamma - 2)
    Y = np.zeros(n)
    for k in range(gamma + 2, n):
        Y[k] = (a1 * Y[k - 1] + a2 * Y[k - 2]
                + b1 * U[k - 1 - gamma, 0]
                + b2 * U[k - 2 - gamma, 0])
    return T, U, Y


def _make_integrating_data(n=300, gamma=0, gain=0.42, seed=2):
    """y[k] - y[k-1] = gain*u[k-1-gamma], y[0]=0, u[:gamma+1]=0."""
    rng = np.random.default_rng(seed)
    T = np.arange(n, dtype=float)
    U = np.zeros((n, 1))
    U[gamma + 1:, 0] = rng.normal(size=n - gamma - 1)
    Y = np.zeros(n)
    for k in range(gamma + 1, n):
        Y[k] = Y[k - 1] + gain * U[k - 1 - gamma, 0]
    return T, U, Y


# ---------------------------------------------------------------------
# Contract tests (arx_sim stubbed)
# ---------------------------------------------------------------------

def test_sysid_returns_expected_tuple(captured_arx):
    T, U, Y = _make_arx1_data(n=80)
    out = sysid_MISO(T, U, Y, [1], 1, 0, 0, PREP_RAW)
    assert len(out) == 7
    A, B, SSEtrn, R2trn, SSEval, R2val, Ypred = out
    assert A.shape == (1,)
    assert B.shape == (1, 1)
    for v in (SSEtrn, R2trn, SSEval, R2val):
        assert np.ndim(v) == 0
    assert Ypred.shape == (len(T),)


def test_sysid_forwards_t_and_xs_slices(captured_arx):
    T, U, Y = _make_arx1_data(n=60)
    start_index = 3
    sysid_MISO(T, U, Y, [1], 1, start_index, 0, PREP_RAW)
    args = captured_arx.args
    # args = (A, B, gamma_list, T, Xs, Xinit, Y_init)
    np.testing.assert_array_equal(args[3], T[start_index:])
    np.testing.assert_array_equal(args[4], U[start_index:, :])


def test_sysid_forwards_gamma_list(captured_arx):
    T, U, Y = _make_arx1_data(n=60)
    sysid_MISO(T, U, Y, [1], 1, 0, 0, PREP_RAW)
    assert list(captured_arx.args[2]) == [1]


def test_sysid_forwards_zero_y_init(captured_arx):
    T, U, Y = _make_arx1_data(n=60)
    sysid_MISO(T, U, Y, [1], 1, 0, 0, PREP_RAW)
    assert captured_arx.args[6] == 0.0


def test_sysid_forwards_A_and_B_shapes_for_each_order(captured_arx):
    T, U, Y = _make_arx1_data(n=80)

    sysid_MISO(T, U, Y, [1], 1, 0, 0, PREP_RAW)
    assert captured_arx.args[0].shape == (1,)
    assert captured_arx.args[1].shape == (1, 1)

    sysid_MISO(T, U, Y, [1], 2, 0, 0, PREP_RAW)
    assert captured_arx.args[0].shape == (2,)
    # B has shape (n_inputs, nu) = (1, 2) for 1 input and 2 lags.
    assert captured_arx.args[1].shape == (1, 2)


def test_sysid_forwards_integrating_A_equals_one(captured_arx):
    T, U, Y = _make_integrating_data(n=80)
    sysid_MISO(T, U, Y, [0], 0, 0, 0, PREP_RAW)
    A = captured_arx.args[0]
    assert A.shape == (1,)
    assert A[0] == 1.0


# ---------------------------------------------------------------------
# Recovery tests (real arx_sim)
# ---------------------------------------------------------------------

def test_sysid_first_order_recovers_coefficients():
    T, U, Y = _make_arx1_data(n=300, gamma=1, a1=0.7, b1=0.4, seed=10)
    A, B, *_ = sysid_MISO(T, U, Y, [1], 1, 0, 0, PREP_RAW)
    np.testing.assert_allclose(A[0], 0.7, atol=1e-6)
    np.testing.assert_allclose(B[0, 0], 0.4, atol=1e-6)


def test_sysid_first_order_predictions_match_raw_y():
    T, U, Y = _make_arx1_data(n=300, gamma=1, a1=0.7, b1=0.4, seed=11)
    *_, Ypred = sysid_MISO(T, U, Y, [1], 1, 0, 0, PREP_RAW)
    # Warm-up samples reproduce Y (both are zero there).
    np.testing.assert_allclose(Ypred, Y, atol=1e-6)


def test_sysid_second_order_recovers_coefficients():
    T, U, Y = _make_arx2_data(n=400, gamma=1,
                              a1=0.7, a2=0.01, b1=0.4, b2=0.05, seed=12)
    A, B, *_ = sysid_MISO(T, U, Y, [1], 2, 0, 0, PREP_RAW)

    # Coefficient ordering (see sysid_MISO):
    #   A = [a2, a1]        (oldest lag first)
    #   B = [[b2, b1]]      (oldest lag first)
    np.testing.assert_allclose(A[0], 0.01, atol=1e-4)   # a2
    np.testing.assert_allclose(A[1], 0.7, atol=1e-4)   # a1
    np.testing.assert_allclose(B[0, 0], 0.05, atol=1e-4)  # b2
    np.testing.assert_allclose(B[0, 1], 0.40, atol=1e-4)  # b1


def test_sysid_integrating_recovers_gain():
    T, U, Y = _make_integrating_data(n=300, gamma=0, gain=0.42, seed=13)
    A, B, *_ = sysid_MISO(T, U, Y, [0], 0, 0, 0, PREP_RAW)
    assert A[0] == 1.0
    np.testing.assert_allclose(B[0, 0], 0.42, atol=1e-6)


def test_sysid_runs_with_prep_raw():
    T, U, Y = _make_arx1_data(n=200)
    A, B, SSEtrn, R2trn, SSEval, R2val, Ypred = sysid_MISO(
        T, U, Y, [1], 1, 0, 0, PREP_RAW,
    )
    assert np.all(np.isfinite(A))
    assert np.all(np.isfinite(B))
    assert np.isfinite(SSEtrn) and np.isfinite(R2trn)
    assert np.isfinite(SSEval) and np.isfinite(R2val)
    assert Ypred.shape == (len(T),)


@pytest.mark.parametrize("mode", [PREP_REMOVE_MEAN, PREP_SUBTRACT_INITIAL])
def test_sysid_runs_with_other_prep_modes(mode):
    # start_index=1 keeps the initial-sample subtraction well defined.
    T, U, Y = _make_arx1_data(n=200)
    A, B, *_ = sysid_MISO(T, U, Y, [1], 1, 1, 0, mode)
    assert A.shape == (1,)
    assert B.shape == (1, 1)


def test_sysid_validation_split_shapes(captured_arx):
    T, U, Y = _make_arx1_data(n=100)
    A, B, SSEtrn, R2trn, SSEval, R2val, Ypred = sysid_MISO(
        T, U, Y, [1], 1, start_index=0, yvalidstart=60, data_prep_stat=PREP_RAW,
    )
    # The stub returns zeros of length len(T[start_index:]) = 100.
    assert Ypred.shape == (100,)
    assert np.ndim(SSEtrn) == 0 and np.ndim(SSEval) == 0
    assert np.ndim(R2trn) == 0 and np.ndim(R2val) == 0


if __name__ == "__main__":
    pytest.main([__file__])