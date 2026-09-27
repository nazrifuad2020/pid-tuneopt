import numpy as np

from pid_tuneopt.identification.least_squares.utils import (
    _prepend_constant,
    _apply_time_delay,
)


class TestPrependConstant:
    def test_zero_count_returns_original(self):
        v = np.array([1.0, 2.0, 3.0])
        out = _prepend_constant(v, 0, 9.0)
        np.testing.assert_array_equal(out, v)

    def test_negative_count_returns_original(self):
        v = np.array([1.0, 2.0, 3.0])
        out = _prepend_constant(v, -3, 9.0)
        np.testing.assert_array_equal(out, v)

    def test_positive_count_prepends_fill(self):
        v = np.array([1.0, 2.0])
        out = _prepend_constant(v, 3, 0.5)
        np.testing.assert_array_equal(out, [0.5, 0.5, 0.5, 1.0, 2.0])

    def test_preserves_dtype(self):
        v = np.array([1, 2, 3], dtype=np.int64)
        out = _prepend_constant(v, 2, 0)
        assert out.dtype == v.dtype


class TestApplyTimeDelay:
    def test_zero_delay_returns_input(self):
        t = np.arange(5, dtype=float)
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        out = _apply_time_delay(t, y, delay=0.0, sample_time=1.0)
        np.testing.assert_array_equal(out, y)

    def test_single_sample_delay(self):
        t = np.arange(5, dtype=float)
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        out = _apply_time_delay(t, y, delay=1.0, sample_time=1.0)
        # y is shifted right by 1 and zero-padded at the start
        np.testing.assert_allclose(out, [0.0, 0.0, 1.0, 2.0, 3.0])

    def test_two_sample_delay(self):
        t = np.arange(5, dtype=float)
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        out = _apply_time_delay(t, y, delay=2.0, sample_time=1.0)
        np.testing.assert_allclose(out, [0.0, 0.0, 0.0, 1.0, 2.0])

    def test_output_length_preserved(self):
        t = np.linspace(0, 1, 11)
        y = np.sin(t)
        out = _apply_time_delay(t, y, delay=0.3, sample_time=0.1)
        assert out.shape == y.shape