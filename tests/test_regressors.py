import numpy as np

from pid_tuneopt.identification.least_squares.regressors import (
    _delayed_input_column,
    _build_input_regressor_matrix,
)


X = np.array([
    [1.0, 10.0],
    [2.0, 20.0],
    [3.0, 30.0],
    [4.0, 40.0],
    [5.0, 50.0],
])


class TestDelayedInputColumn:
    def test_gamma_equals_mingamma_no_pad(self):
        col = _delayed_input_column(X, gamma=1, mingamma=1, init_value=0.0, col_idx=0)
        np.testing.assert_array_equal(col, [1.0, 2.0, 3.0])

    def test_gamma_zero_equals_mingamma(self):
        col = _delayed_input_column(X, gamma=0, mingamma=0, init_value=0.0, col_idx=0)
        np.testing.assert_array_equal(col, [1.0, 2.0, 3.0, 4.0])

    def test_gamma_greater_than_mingamma_prepends_constant(self):
        col = _delayed_input_column(X, gamma=2, mingamma=1, init_value=99.0, col_idx=0)
        np.testing.assert_array_equal(col, [99.0, 1.0, 2.0])

    def test_other_column_selected(self):
        col = _delayed_input_column(X, gamma=1, mingamma=1, init_value=0.0, col_idx=1)
        np.testing.assert_array_equal(col, [10.0, 20.0, 30.0])

    def test_returns_1d(self):
        col = _delayed_input_column(X, gamma=1, mingamma=1, init_value=0.0, col_idx=0)
        assert col.ndim == 1


class TestBuildInputRegressorMatrix:
    def test_single_input(self):
        init_row = np.zeros((1, 1))
        M = _build_input_regressor_matrix(X[:, :1], [1], mingamma=1, init_row=init_row)
        assert M.shape == (3, 1)
        np.testing.assert_array_equal(M[:, 0], [1.0, 2.0, 3.0])

    def test_two_inputs_with_mixed_gamma(self):
        init_row = np.zeros((1, 2))
        M = _build_input_regressor_matrix(X, [1, 2], mingamma=1, init_row=init_row)
        assert M.shape == (3, 2)
        np.testing.assert_array_equal(M[:, 0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(M[:, 1], [0.0, 10.0, 20.0])

    def test_all_columns_same_length(self):
        init_row = np.zeros((1, 2))
        M = _build_input_regressor_matrix(X, [0, 2], mingamma=0, init_row=init_row)
        assert M.shape[0] == 4