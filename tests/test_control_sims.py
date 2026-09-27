# -*- coding: utf-8 -*-
"""Unit tests for :mod:`pid_tuneopt.controls.sims`.

Run with::

    pytest tests/test_sims.py -v

If your package layout differs, adjust the import line below.
"""

import numpy as np
import pytest
 
from pid_tuneopt.controls import sims


# ===========================================================================
# Helpers / fixtures
# ===========================================================================

def _discrete_plant(A, B, thetap=0.0):
    """Model dict accepted by ``set_point_tracking`` / ``disturbance_rejection``."""
    Kp = np.sum(np.array(B)) / (1 + np.sum(-1*np.array(A)))
    return {"A": np.asarray(A, dtype=float),
            "B": np.asarray(B, dtype=float),
            "thetap": float(thetap),
            "Kp": Kp,
        }


# ===========================================================================
# set_point_tracking
# ===========================================================================

class TestSetPointTracking:
    """Set-point tracking simulation."""

    def test_gotoendtime_without_endtime_returns_error(self):
        model = _discrete_plant([0.5], [0.5])
        result = sims.set_point_tracking(model, 1.0, 1.0, 0.0, gotoendtime=1)
        assert result == ([], [], -1)

    def test_missing_AB_returns_none(self):
        model = {"thetap": 0.0}
        result = sims.set_point_tracking(model, 1.0, 1.0, 0.0)
        assert result is None

    def test_output_lengths_match(self):
        model = _discrete_plant([0.5], [0.5])
        u, y, status = sims.set_point_tracking(
            model, 0.3, 10.0, 0.0, Ts=1.0, spmag=1.0, endtime=50)
        assert len(u) == len(y)
        assert len(y) > 0
        assert status in (0, 1, 2)

    def test_control_respects_limits(self):
        model = _discrete_plant([0.5], [0.5])
        u, y, _ = sims.set_point_tracking(
            model, 5.0, 1.0, 0.0, umin=0.0, umax=2.0,
            Ts=1.0, spmag=1.0, endtime=50)
        assert np.all(u <= 2.0 + 1e-12)
        assert np.all(u >= 0.0 - 1e-12)

    def test_response_moves_toward_setpoint(self):
        model = _discrete_plant([0.5], [0.5])
        u, y, status = sims.set_point_tracking(
            model, 0.3, 1.0, 0.0, Ts=1.0, spmag=1.0, endtime=200)
        # Output should be rising toward the set-point (1.0).
        assert y[-1] > 0.5
        assert y[-1] <= 1.0 + 1e-3

    def test_derivative_on_measurement_runs(self):
        model = _discrete_plant([0.5], [0.5])
        u, y, status = sims.set_point_tracking(
            model, 0.3, 1.0, 0.1, D_on_Pv=1,
            Ts=1.0, spmag=1.0, endtime=30)
        assert len(u) == len(y)
        assert status in (0, 1, 2)


# ===========================================================================
# disturbance_rejection
# ===========================================================================

class TestDisturbanceRejection:
    """Load-disturbance rejection simulation."""

    def test_output_lengths_match(self):
        model = _discrete_plant([0.5], [0.5], thetap=1.0)
        dm = _discrete_plant([0.9], [0.1])
        u, y, status = sims.disturbance_rejection(
            model, dm, 0.3, 1.0, 0.0,
            Ts=1.0, idistmag=1.0, endtime=15)
        assert len(u) == len(y) > 0
        assert status in (0, 1, 2)    

    def test_missing_AB_returns_none(self):
        model = _discrete_plant([0.5], [0.5])
        dm = {}
        result = sims.disturbance_rejection(
                model, dm, 1.0, 1.0, 0.0)
        assert result is None

    def test_gotoendtime_without_endtime_returns_error(self):
        model = _discrete_plant([0.5], [0.5])
        dm = _discrete_plant([0.9], [0.1])
        result = sims.disturbance_rejection(
            model, dm, 1.0, 1.0, 0.0, gotoendtime=1)
        assert result == ([], [], -1)


if __name__ == "__main__":
    pytest.main([__file__])