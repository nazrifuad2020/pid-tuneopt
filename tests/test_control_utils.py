# -*- coding: utf-8 -*-
"""Unit tests for :mod:`pid_tuneopt.controls.utils`.

Run with::

    pytest tests/test_control_utils.py -v

If your package layout differs, adjust the import line below.
"""
import math 
import numpy as np
from pid_tuneopt.controls import utils

import pytest


# ===========================================================================
# Helpers / fixtures
# ===========================================================================

def _cont_model(num, den, thetap=0.0):
    """Continuous-time model dict (``num``/``den``) for the fallback path."""
    return {"num": list(num), "den": list(den), "thetap": float(thetap), "Kp": list(num)[-1]/list(den)[-1]}


# ===========================================================================
# find_peaks_wrapper
# ===========================================================================

class TestFindPeaks:
    """Peak detection."""

    def test_sine_wave_detects_extrema(self):
        # One period of sin over [0, 2*pi]; expect one maximum and one minimum.
        t = np.linspace(0.0, 2.0 * np.pi, 1001)
        y = np.sin(t)
        peaks = utils.find_peaks_wrapper(y)
        assert len(peaks) == 2
        assert peaks[0] == pytest.approx(250, abs=2)   # max at t = pi/2
        assert peaks[1] == pytest.approx(750, abs=2)   # min at t = 3*pi/2

    def test_all_zeros_returns_empty(self):
        peaks = utils.find_peaks_wrapper(np.zeros(100))
        assert peaks == []

    def test_constant_nonzero_returns_empty(self):
        peaks = utils.find_peaks_wrapper(np.full(100, 3.14))
        assert peaks == []

    def test_monotonic_increasing_returns_empty(self):
        peaks = utils.find_peaks_wrapper(np.arange(50, dtype=float) + 1.0)
        assert peaks == []

    def test_monotonic_decreasing_returns_empty(self):
        peaks = utils.find_peaks_wrapper(-(np.arange(50, dtype=float) + 1.0))
        assert peaks == []

    def test_fifo_keeps_only_last_four_peaks(self):
        # 3 full periods -> 6 extrema.  The buffer must keep only the last 4.
        # Sample index of time t is  i = t * 6000 / (6*pi) = t * 1000 / pi.
        # Extrema at pi/2 + k*pi  ->  indices 500, 1500, 2500, 3500, 4500, 5500.
        t = np.linspace(0.0, 6.0 * np.pi, 6001)
        y = np.sin(t)
        peaks = utils.find_peaks_wrapper(y)
        assert len(peaks) == 4
        # The FIFO must drop the first two and keep the last four.
        expected = [2500, 3500, 4500, 5500]
        assert peaks == pytest.approx(expected, abs=2)

    def test_single_peak_returns_one_index(self):
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0])
        peaks = utils.find_peaks_wrapper(y)
        assert peaks == [4]


# ===========================================================================
# find_settling_time
# ===========================================================================

class TestFindSettlingTime:
    """Settling-time extraction."""

    def test_exponential_settling(self):
        # y(t) = 1 - 0.1 * exp(-t)  ->  err = 0.1 * exp(-t).
        # err crosses 0.05 at t = ln(2) ~ 0.6931.
        t = np.linspace(0.0, 10.0, 10001)
        y = 1.0 - 0.1 * np.exp(-t)
        ts = utils.find_settling_time(t, y, spmag=1.0)
        assert ts == pytest.approx(math.log(2.0), abs=0.01)

    def test_never_settles_returns_inf(self):
        t = np.linspace(0.0, 10.0, 101)
        y = 0.5 * np.ones_like(t)   # error 0.5 > 5 %
        ts = utils.find_settling_time(t, y, spmag=1.0)
        assert math.isinf(ts)

    def test_always_in_band_returns_inf(self):
        # Note: the original implementation returns ``inf`` when the
        # response never leaves the 5 % band (no crossing to interpolate).
        t = np.linspace(0.0, 10.0, 101)
        y = 0.99 * np.ones_like(t)
        ts = utils.find_settling_time(t, y, spmag=1.0)
        assert math.isinf(ts)

    def test_returns_float(self):
        t = np.linspace(0.0, 10.0, 1001)
        y = 1.0 - 0.5 * np.exp(-t)
        ts = utils.find_settling_time(t, y, spmag=1.0)
        assert isinstance(ts, float)


# ===========================================================================
# stability_analysis
# ===========================================================================

class TestStabilityAnalysis:
    """Gain / phase margin computation."""

    def test_returns_three_finite_floats(self):
        model = _cont_model([1.0], [1.0, 1.0], thetap=0.0)
        GM, PM, max_delay = utils.stability_analysis(1.0, 1.0, 0.1, model)
        assert isinstance(GM, float)
        assert isinstance(PM, float)
        assert isinstance(max_delay, float)
        assert np.isfinite(GM)
        assert np.isfinite(PM)
        assert np.isfinite(max_delay)

    def test_phase_margin_degrades_with_gain(self):
        # Second-order plant: its open-loop phase lives in (0, -180) deg,
        # so scipy's phase wrapping never kicks in and PM decreases
        # monotonically as the loop gain grows.
        model = _cont_model([1.0], [1.0, 2.0, 1.0], thetap=0.0)
        _, PM_low, _ = utils.stability_analysis(0.2, 1.0, 0.0, model)
        _, PM_high, _ = utils.stability_analysis(2.0, 1.0, 0.0, model)
        assert PM_high < PM_low
        # Both margins must live in the physically meaningful range.
        assert 0.0 < PM_high < 180.0
        assert 0.0 < PM_low  < 180.0

    def test_gain_margin_is_positive(self):
        model = _cont_model([1.0], [1.0, 3.0, 3.0, 1.0], thetap=0.0)
        GM, _, _ = utils.stability_analysis(0.5, 1.0, 0.0, model)
        assert GM > 0


if __name__ == "__main__":
    pytest.main([__file__])
