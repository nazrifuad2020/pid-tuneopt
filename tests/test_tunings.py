# -*- coding: utf-8 -*-
"""Unit tests for :mod:`pid_tuneopt.controls.tunings`.

Run with::

    pytest tests/test_tunings.py -v

If your package layout differs, adjust the import line below.
"""

import numpy as np
import pytest
 
from pid_tuneopt.controls import tunings


# ===========================================================================
# Helpers / fixtures
# ===========================================================================

def _fopdt(Kp=1.0, thetap=0.0):
    """Minimal FOPDT dict accepted by ``imc_tuner`` (no ``taup`` key)."""
    return {"Kp": float(Kp), "thetap": float(thetap)}


def _sopdt(Kp=1.0, taup=1.0, zetap=0.7, tauz=0.0, thetap=0.0):
    """SOPDT dict accepted by ``imc_tuner`` (``taup`` key present)."""
    return {"Kp": float(Kp), "taup": float(taup), "zetap": float(zetap),
            "tauz": float(tauz), "thetap": float(thetap)}


# ===========================================================================
# imc_tuner
# ===========================================================================

class TestImcTuner:
    """IMC tuning rules."""

    def test_fopdt_exact_pid_values(self):
        # theta = thetap + Ts/2 = 0 + 0.5 = 0.5
        # Kc = (2*1 + 0.5) / (1 * (1 + 0.25)^2) = 1.6
        # Ti = 2 + 0.5 = 2.5
        # Td = (1*0.5 + 0.25) / 2.5 = 0.225
        Kc, Ti, Td = tunings.imc_tuner(_fopdt(Kp=1.0, thetap=0.0),
                                    tauc=1.0, pid_type=0, Ts=1.0)
        assert Kc == pytest.approx(1.6)
        assert Ti == pytest.approx(2.5)
        assert Td == pytest.approx(0.225)

    def test_fopdt_exact_pi_values(self):
        # pid_type=1: Kc = (2*1 + 0.5) / (1 * (1 + 0.5)^2) = 2.5/2.25
        #             Ti = 2.5, Td = 0
        Kc, Ti, Td = tunings.imc_tuner(_fopdt(Kp=1.0, thetap=0.0),
                                    tauc=1.0, pid_type=1, Ts=1.0)
        assert Kc == pytest.approx(2.5 / 2.25)
        assert Ti == pytest.approx(2.5)
        assert Td == 0.0

    def test_returns_three_positive_values(self):
        Kc, Ti, Td = tunings.imc_tuner(_fopdt(Kp=2.0, thetap=1.0), tauc=1.0)
        assert Kc > 0
        assert Ti > 0
        assert Td >= 0

    def test_higher_tauc_reduces_gain(self):
        Kc_fast, _, _ = tunings.imc_tuner(_fopdt(thetap=0.5), tauc=0.5)
        Kc_slow, _, _ = tunings.imc_tuner(_fopdt(thetap=0.5), tauc=2.0)
        assert Kc_fast > Kc_slow

    def test_sopdt_undamped_pi(self):
        # zetap == 0, pid_type=1
        Kc, Ti, Td = tunings.imc_tuner(
            _sopdt(taup=1.0, zetap=0.0, tauz=0.0, thetap=0.5),
            tauc=1.0, pid_type=1)
        assert Kc > 0 and Ti > 0 and Td == 0.0

    def test_sopdt_undamped_pid(self):
        # zetap == 0, pid_type=0
        Kc, Ti, Td = tunings.imc_tuner(
            _sopdt(taup=1.0, zetap=0.0, tauz=0.0, thetap=0.5),
            tauc=1.0, pid_type=0)
        assert Kc > 0 and Ti > 0 and Td > 0

    def test_sopdt_no_lead(self):
        # zetap > 0, tauz == 0
        Kc, Ti, Td = tunings.imc_tuner(
            _sopdt(taup=2.0, zetap=0.8, tauz=0.0, thetap=0.5), tauc=1.0)
        assert Kc > 0 and Ti > 0 and Td > 0

    def test_sopdt_with_lead(self):
        # zetap > 0, tauz > 0
        Kc, Ti, Td = tunings.imc_tuner(
            _sopdt(taup=2.0, zetap=0.8, tauz=0.5, thetap=0.5), tauc=1.0)
        assert Kc > 0 and Ti > 0 and Td >= 0

    def test_sopdt_with_inverse_lead(self):
        # zetap > 0, tauz < 0
        Kc, Ti, Td = tunings.imc_tuner(
            _sopdt(taup=2.0, zetap=0.8, tauz=-0.5, thetap=0.5), tauc=1.0)
        assert Kc > 0 and Ti > 0 and Td > 0