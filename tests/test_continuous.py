import numpy as np
import pytest
from scipy import signal

from pid_tuneopt.identification.least_squares.continuous import (
    d2c,
    tustin_approx_d2c,
    _continuous_tf_from_params,
    _discrete_tf_coeffs,
)


class TestContinuousTfFromParams:
    @staticmethod
    def _assert_tf_equal(tf, num, den):
        """Compare two TF objects after applying SciPy's normalisation."""
        ref = signal.TransferFunction(num, den)
        np.testing.assert_allclose(tf.num, ref.num, atol=1e-12)
        np.testing.assert_allclose(tf.den, ref.den, atol=1e-12)

    def test_first_order(self):
        tf = _continuous_tf_from_params(
            Kp=2.0, time_constant=1.5,
            damping_coeff=0.0, time_zero=0.0, norder=1,
        )
        self._assert_tf_equal(tf, num=[2.0], den=[1.5, 1.0])

    def test_second_order(self):
        tf = _continuous_tf_from_params(
            Kp=3.0, time_constant=0.5,
            damping_coeff=0.7, time_zero=0.0, norder=2,
        )
        self._assert_tf_equal(tf, num=[3.0], den=[0.25, 0.7, 1.0])

    def test_third_order(self):
        tf = _continuous_tf_from_params(
            Kp=1.0, time_constant=2.0,
            damping_coeff=0.3, time_zero=0.5, norder=3,
        )
        self._assert_tf_equal(tf, num=[0.5, 1.0], den=[4.0, 1.2, 1.0])


class TestDiscreteTfCoeffs:
    def test_first_order(self):
        sysdisc = signal.TransferFunction([1.0], [1.0, -0.5], dt=1.0)
        A, B = _discrete_tf_coeffs(sysdisc)
        np.testing.assert_allclose(A, [0.5])
        np.testing.assert_allclose(B, [1.0])

    def test_second_order(self):
        sysdisc = signal.TransferFunction([1.0, 0.5], [1.0, -0.3, 0.1], dt=1.0)
        A, B = _discrete_tf_coeffs(sysdisc)
        np.testing.assert_allclose(A, [-0.1, 0.3])
        np.testing.assert_allclose(B, [0.5, 1.0])


class TestTustin:
    def test_recovers_first_order_from_tustin_discretisation(self):
        # H(s) = 1 / (s + 1), Ts = 0.1, discretised via Tustin:
        #   H(z) = (z + 1) / (21 z - 19)
        nums, dens = tustin_approx_d2c([1.0, 1.0], [21.0, -19.0], 0.1)
        # Normalise so dens[-1] == 1
        nums = np.atleast_1d(nums)
        dens = np.atleast_1d(dens)
        nums_norm = nums / dens[-1] * dens[-1]  # no-op, keep API explicit
        # Compare shapes and ratio: nums/dens ~ K/(tau*s+1)
        # After the substitution, dens[0] / dens[1] should be tau=1
        # and nums should be proportional to dens.
        ratio = nums[-1] / dens[-1]
        assert np.isclose(dens[0] / dens[-1], 1.0, atol=1e-10)
        assert np.isclose(ratio, 1.0, atol=1e-10)

    def test_constant_denominator(self):
        # If numerator is proportional to denominator, result is constant
        nums, dens = tustin_approx_d2c([1.0, 0.5], [1.0, 0.5], 0.1)
        # dens normalised, nums should be proportional to dens
        # Ratio nums[-1]/dens[-1] should equal ratio nums[0]/dens[0]
        assert np.isclose(nums[0] / dens[0], nums[-1] / dens[-1], atol=1e-10)

    def test_zero_gain(self):
        nums, dens = tustin_approx_d2c([0.0, 0.0], [1.0, -0.5], 0.1)
        assert np.allclose(nums, 0.0)


class TestD2CIntegrating:
    def test_returns_expected_dict(self):
        A = np.array([0.5])
        B = np.array([[0.5]])
        result = d2c(A, B, gamma_list=[1], model_order=0, sample_time=1.0)
        assert isinstance(result, list) and len(result) == 1
        r = result[0]
        assert r["Method"] == "ZOH"
        assert r["BIBO_stable"] is False
        assert np.isclose(r["thetap"], 1.0)
        assert np.isclose(r["Kp"], 0.5)
        np.testing.assert_allclose(r["den"], [1, 0])


class TestD2CFirstOrderStable:
    def test_zoh_roundtrip_recovers_continuous_params(self):
        K, tau, Ts = 2.0, 1.0, 0.5
        p = np.exp(-Ts / tau)
        b1 = K * (1.0 - p)
        A = np.array([p])
        B = np.array([[b1]])
        result = d2c(A, B, gamma_list=[1], model_order=1, sample_time=Ts)
        assert len(result) == 1
        r = result[0]
        assert r["BIBO_stable"] is True
        assert r["Method"] == "ZOH"
        assert np.isclose(r["Kp"], K, atol=1e-8)
        assert np.isclose(r["taup"], tau, atol=1e-8)
        assert np.isclose(r["thetap"], Ts, atol=1e-12)

    def test_time_delay_scales_with_sample_time(self):
        K, tau, Ts = 1.0, 0.5, 0.1
        p = np.exp(-Ts / tau)
        b1 = K * (1.0 - p)
        result = d2c(np.array([p]), np.array([[b1]]),
                     gamma_list=[3], model_order=1, sample_time=Ts)
        assert np.isclose(result[0]["thetap"], 3 * Ts)


class TestD2CSecondOrderStable:
    def test_second_order_produces_dict(self):
        # Just a smoke test; use a genuinely second-order stable discrete system
        sysc = signal.TransferFunction([1.0], [1.0, 1.4, 1.0])
        sysd = sysc.to_discrete(dt=0.1, method="zoh")
        a1 = -sysd.den[1]
        a2 = -sysd.den[2]
        b1 = sysd.num[1] if len(sysd.num) > 1 else 0.0
        b2 = sysd.num[2] if len(sysd.num) > 2 else 0.0
        A = np.array([a2, a1])   # note internal ordering
        B = np.array([[b2], [b1]])  # not used directly; API expects (ninput, norder)
        # Reshape B to match API: (ninput, norder)
        B_api = np.array([[b1, b2]])
        result = d2c(A, B_api, gamma_list=[0], model_order=2, sample_time=0.1)
        assert len(result) == 1
        assert "BIBO_stable" in result[0]