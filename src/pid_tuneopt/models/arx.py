# -*- coding: utf-8 -*-
"""
Discrete-time ARX prediction / simulation.
"""

import numpy as np


def arx_sim(A, B, gamma_list, T, Xs, X_inits, Y_init):
    """
    Simulate a discrete-time ARX model.

    Parameters
    ----------
    A : array_like, shape (ny,)
        Autoregressive coefficients.
    B : array_like, shape (ninputs, nu)
        Input (exogenous) coefficients.
    gamma_list : sequence of int, length ninputs
        Per-input time delay (in samples).
    T : array_like
        Time vector. Only ``T.size`` is used.
    Xs : array_like, shape (N, ninputs)
        Input signal history.
    X_inits : array_like, shape (1, ninputs)
        Initial input values used for padding.
    Y_init : float
        Initial output value.

    Returns
    -------
    ypred : ndarray, shape (T.size,)
        Simulated output signal.
    """
    # --- Normalise inputs -------------------------------------------------
    A = np.asarray(A, dtype=float).ravel()
    B = np.asarray(B, dtype=float)
    Xs = np.asarray(Xs, dtype=float)
    X_inits = np.atleast_2d(np.asarray(X_inits, dtype=float))
    T = np.asarray(T)

    ninputs = Xs.shape[1]
    ny = A.shape[0]
    nu = B.shape[1]
    mingamma = min(gamma_list)

    # --- Build input-history matrix (one column per input) ----------------
    Xdot_cols = []
    for i in range(ninputs):
        gamma = gamma_list[i]
        col = Xs[:-1 - gamma, i].flatten()

        # Pad short histories so every column has the same length.
        if mingamma - gamma < 0:
            pad = np.full(gamma - mingamma, X_inits[0, i], dtype=float)
            col = np.concatenate((pad, col))

        # Prepend nu-1 initial inputs (input lags).
        if nu > 1:
            pad = np.full(nu - 1, X_inits[0, i], dtype=float)
            col = np.concatenate((pad, col))

        Xdot_cols.append(col)

    Xdotmatr = np.column_stack(Xdot_cols)

    # --- Simulation -------------------------------------------------------
    ypred = np.zeros(T.size, dtype=float)
    ybefore = np.full(ny, Y_init, dtype=float)

    # Warm-up region: the first (mingamma + 1) samples equal Y_init.
    ypred[:mingamma + 1] = Y_init

    k = mingamma + 1
    m = nu
    while k < ypred.size:
        # Autoregressive part.
        ypred[k] = A @ ybefore

        # Exogenous part.
        for i in range(ninputs):
            for j in range(nu):
                ypred[k] += B[i, j] * Xdotmatr[m - nu + j, i]

        # Shift output history.
        for j in range(ny - 1):
            ybefore[j] = ybefore[j + 1]
        ybefore[ny - 1] = ypred[k]

        k += 1
        m += 1

    return ypred
