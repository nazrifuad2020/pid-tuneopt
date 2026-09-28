from __future__ import annotations

import numpy as np

from ...models.arx import arx_sim
from .fits import (
    linear_least_squares_fit_first_order,
    linear_least_squares_fit_integrating,
    linear_least_squares_fit_second_order,
)
from .metrics import _compute_fit_metrics
from .preprocessing import generate_training_data, _initial_input_reference


def sysid_MISO(
    Traw,
    Xraws,
    Yraw,
    gamma_list,
    model_order: int,
    start_index: int,
    yvalidstart: int,
    data_prep_stat: int,
):
    """Identify a discrete-time MISO ARX model via least squares."""
    (Xdots, Ydot, _Ttrain, Xtrains, Ytrain, _Yval,
     _Xrefs, Yref) = generate_training_data(
        Traw, Xraws, Yraw, start_index, yvalidstart, data_proprocess=data_prep_stat)

    Xinit = _initial_input_reference(Xraws, start_index, data_prep_stat)
    Yinit = Ytrain[0]

    Ytrain_centered = Ytrain - Ytrain[0]
    if model_order == 1:
        a1, b1 = linear_least_squares_fit_first_order(
            Xtrains, Ytrain_centered, gamma_list, Xinit)
        A = np.array([a1])
        B = np.matrix([b1]).transpose()
    elif model_order == 2:
        a1, a2, b1, b2 = linear_least_squares_fit_second_order(
            Xtrains, Ytrain_centered, gamma_list, Xinit)
        A = np.array([a2, a1])
        B = np.matrix([b2, b1]).transpose()
    else:
        a1, b1 = linear_least_squares_fit_integrating(
            Xtrains, Ytrain, gamma_list, Xinit)
        A = np.array([a1])
        B = np.matrix([b1]).transpose()

    T = Traw[start_index:]
    Ypred = arx_sim(A, B, gamma_list, T, Xdots, Xinit, 0.0)
    Ypred += (Yinit + Yref)

    SSEtrain, R2_train, SSEval, R2_val = _compute_fit_metrics(
        Yraw, Ypred, start_index, yvalidstart)
    return A, B, SSEtrain, R2_train, SSEval, R2_val, Ypred