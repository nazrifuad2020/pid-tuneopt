from __future__ import annotations

import numpy as np

from .linalg import _solve_normal_equations
from .regressors import _build_input_regressor_matrix, _delayed_input_column


def linear_least_squares_fit_integrating(Xdot, Ydot, gamma_list, Xinit):
    """Fit an integrating (pure-gain + delay) MISO model."""
    mingamma = int(min(gamma_list))
    Umatrix = _build_input_regressor_matrix(Xdot, gamma_list, mingamma, Xinit)
    z = Ydot[mingamma + 1:] - Ydot[mingamma:-1]
    xsol = _solve_normal_equations(Umatrix, z)
    return 1.0, [xsol[i] for i in range(Xdot.shape[1])]


def linear_least_squares_fit_first_order(Xdot, Ydot, gamma_list, Xinit):
    """Fit a first-order ARX MISO model."""
    mingamma = int(min(gamma_list))
    Umatrix = np.column_stack((
        Ydot[mingamma:-1],
        _build_input_regressor_matrix(Xdot, gamma_list, mingamma, Xinit),
    ))
    z = Ydot[mingamma + 1:]
    xsol = _solve_normal_equations(Umatrix, z)
    return xsol[0], [xsol[i + 1] for i in range(Xdot.shape[1])]


def linear_least_squares_fit_second_order(Xdot, Ydot, gamma_list, Xinit):
    """Fit a second-order ARX MISO model."""
    ninputs = Xdot.shape[1]
    mingamma = int(min(gamma_list))

    col1 = Ydot[mingamma:-1]
    if mingamma > 0:
        col2 = Ydot[mingamma - 1:-2]
    else:
        col2 = np.append(Ydot[0], col1[:-1])

    columns = [col1, col2]
    for i in range(ninputs):
        col3 = _delayed_input_column(Xdot, int(gamma_list[i]), mingamma, Xinit[0, i], i)
        col4 = np.append(Xinit[0, i], col3[:-1])
        columns.extend([col3, col4])

    Umatrix = np.column_stack(columns)
    z = Ydot[mingamma + 1:]
    xsol = _solve_normal_equations(Umatrix, z)

    b1 = [xsol[2 * (i + 1)] for i in range(ninputs)]
    b2 = [xsol[2 * (i + 1) + 1] for i in range(ninputs)]
    return xsol[0], xsol[1], b1, b2
