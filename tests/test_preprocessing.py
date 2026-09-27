import numpy as np
import pytest

from pid_tuneopt.identification.least_squares.constants import (
    PREP_SUBTRACT_INITIAL,
    PREP_REMOVE_MEAN,
    PREP_RAW,
)
from pid_tuneopt.identification.least_squares.preprocessing import (
    find_start_index,
    generate_training_data,
    _initial_input_reference,
)


class TestFindStartIndex:
    def test_returns_zero_for_non_subtract_mode(self):
        U = np.arange(10, dtype=float).reshape(-1, 1)
        assert find_start_index(U, data_proprocess=PREP_REMOVE_MEAN) == 0
        assert find_start_index(U, data_proprocess=PREP_RAW) == 0

    def test_detects_first_deviation(self):
        U = np.ones((10, 1))
        U[3:] = 2.0
        assert find_start_index(U, data_proprocess=PREP_SUBTRACT_INITIAL) == 3

    def test_uses_minimum_across_inputs(self):
        U = np.ones((10, 2))
        U[5:, 0] = 2.0
        U[2:, 1] = 3.0
        assert find_start_index(U, data_proprocess=PREP_SUBTRACT_INITIAL) == 2

    def test_no_change_prints_and_returns_minus_one(self, capsys):
        U = np.ones((10, 1))
        result = find_start_index(U, data_proprocess=PREP_SUBTRACT_INITIAL)
        assert result == -1
        captured = capsys.readouterr()
        assert "No input signal is injected" in captured.out


class TestGenerateTrainingData:
    def test_raw_mode(self):
        T = np.arange(6, dtype=float)
        X = np.arange(12, dtype=float).reshape(6, 2)
        Y = np.arange(6, dtype=float)
        out = generate_training_data(T, X, Y, start_index=1, yvalidstart=0,
                                     data_proprocess=PREP_RAW)
        Xdots, Ydot, Ttrain, Xtrains, Ytrain, Yval, Xrefs, Yref = out
        np.testing.assert_array_equal(Xdots, X[1:])
        np.testing.assert_array_equal(Ydot, Y[1:])
        np.testing.assert_array_equal(Xrefs, np.zeros((1, 2)))
        assert Yref == 0.0

    def test_remove_mean_mode(self):
        T = np.arange(6, dtype=float)
        X = np.arange(12, dtype=float).reshape(6, 2)
        Y = np.arange(6, dtype=float)
        out = generate_training_data(T, X, Y, start_index=1, yvalidstart=0,
                                     data_proprocess=PREP_REMOVE_MEAN)
        Xdots, Ydot, *_ = out
        np.testing.assert_allclose(Xdots.mean(axis=0), [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(Ydot.mean(), 0.0, atol=1e-12)

    def test_subtract_initial_mode(self):
        T = np.arange(6, dtype=float)
        X = np.arange(12, dtype=float).reshape(6, 2)
        Y = np.arange(6, dtype=float)
        out = generate_training_data(T, X, Y, start_index=2, yvalidstart=0,
                                     data_proprocess=PREP_SUBTRACT_INITIAL)
        Xdots, Ydot, Ttrain, Xtrains, Ytrain, Yval, Xrefs, Yref = out
        # Xdots = X[2:] - X[1]
        np.testing.assert_array_equal(Xdots, X[2:] - X[1])
        # Ydot = Y[2:] - Y[2]
        np.testing.assert_array_equal(Ydot, Y[2:] - Y[2])
        np.testing.assert_array_equal(Xrefs, X[1])
        assert Yref == Y[2]

    def test_validation_split_shapes(self):
        T = np.arange(20, dtype=float)
        X = np.zeros((20, 1))
        Y = np.arange(20, dtype=float)
        out = generate_training_data(T, X, Y, start_index=2, yvalidstart=10,
                                     data_proprocess=PREP_RAW)
        _, _, Ttrain, Xtrains, Ytrain, Yval, _, _ = out
        assert len(Ttrain) == 9       # Traw[2:11]
        assert Xtrains.shape[0] == 9
        assert Ytrain.shape[0] == 9
        np.testing.assert_array_equal(Yval, Y[10:])

    def test_no_validation_split(self):
        T = np.arange(10, dtype=float)
        X = np.zeros((10, 1))
        Y = np.arange(10, dtype=float)
        out = generate_training_data(T, X, Y, start_index=2, yvalidstart=0,
                                     data_proprocess=PREP_RAW)
        _, _, _, Xtrains, Ytrain, Yval, _, _ = out
        assert Xtrains.shape[0] == 8
        assert Ytrain.shape[0] == 8
        assert Yval == []


class TestInitialInputReference:
    def test_raw_returns_mean(self):
        Xraws = np.arange(20, dtype=float).reshape(10, 2)
        ref = _initial_input_reference(Xraws, start_index=2, data_prep_stat=PREP_RAW)
        np.testing.assert_allclose(ref, Xraws[2:].mean(axis=0).reshape(1, -1))

    def test_others_return_zero_row(self):
        Xraws = np.arange(20, dtype=float).reshape(10, 2)
        for mode in (PREP_SUBTRACT_INITIAL, PREP_REMOVE_MEAN):
            ref = _initial_input_reference(Xraws, start_index=2, data_prep_stat=mode)
            np.testing.assert_array_equal(ref, np.zeros((1, 2)))