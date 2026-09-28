from __future__ import annotations

import numpy as np

from .constants import PREP_SUBTRACT_INITIAL, PREP_REMOVE_MEAN, PREP_RAW


def find_start_index(Uraw: np.ndarray, data_proprocess: int = 0) -> int:
    """Locate the first sample at which any input deviates from its initial value."""
    if data_proprocess != PREP_SUBTRACT_INITIAL:
        return 0

    start_indexes = []
    for i in range(Uraw.shape[1]):
        changes = np.argwhere(np.abs(Uraw[:, i] - Uraw[0, i]) > 0)
        if changes.shape[0] > 0:
            start_indexes.append(int(changes[0, 0]))

    if not start_indexes:
        print("No input signal is injected")
        return -1
    return min(start_indexes)


def generate_training_data(
    Traw,
    Xraws,
    Yraw,
    start_index: int,
    yvalidstart: int,
    data_proprocess: int = 0,
):
    """Build centred training/validation arrays from the raw I/O records."""
    if data_proprocess == PREP_RAW:
        Xdots = Xraws[start_index:, :]
        Xrefs = np.zeros((1, Xraws.shape[1]))
        Ydot = Yraw[start_index:]
        Yref = 0.0
    elif data_proprocess == PREP_REMOVE_MEAN:
        Xrefs = np.mean(Xraws[start_index:, :], axis=0)
        Xdots = Xraws[start_index:, :] - Xrefs
        Yref = np.mean(Yraw[start_index:])
        Ydot = Yraw[start_index:] - Yref
    else:  # PREP_SUBTRACT_INITIAL
        Xdots = Xraws[start_index:, :] - Xraws[start_index - 1, :]
        Xrefs = Xraws[start_index - 1, :]
        Ydot = Yraw[start_index:] - Yraw[start_index]
        Yref = Yraw[start_index]

    if yvalidstart > start_index:
        end = yvalidstart - start_index + 1
        Xtrains = Xdots[:end, :]
        Ytrain = Ydot[:end]
        Ttrain = Traw[start_index:yvalidstart + 1]
        Yval = Yraw[yvalidstart:]
    else:
        Xtrains = Xdots
        Ytrain = Ydot
        Ttrain = Traw[start_index:]
        Yval = []

    return Xdots, Ydot, Ttrain, Xtrains, Ytrain, Yval, Xrefs, Yref


def _initial_input_reference(Xraws, start_index, data_prep_stat):
    """Return the reference input used by the LS / NLP estimators."""
    if data_prep_stat == PREP_RAW:
        return np.mean(Xraws[start_index:, :], axis=0).reshape(1, -1)  # ← 2-D now
    return np.zeros((1, Xraws.shape[1]))