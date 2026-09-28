from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.linalg import inv, logm


def _continuous_tf_from_params(
    Kp: float,
    time_constant: float,
    damping_coeff: float,
    time_zero: float,
    norder: int,
) -> signal.TransferFunction:
    """Build a continuous transfer function for one of the supported orders."""
    if norder == 1:
        return signal.TransferFunction([Kp], [time_constant, 1.0])
    if norder == 2:
        return signal.TransferFunction(
            [Kp],
            [time_constant ** 2, 2 * time_constant * damping_coeff, 1.0],
        )
    # norder == 3: second order with one zero
    return signal.TransferFunction(
        [Kp * time_zero, Kp],
        [time_constant ** 2, 2 * time_constant * damping_coeff, 1.0],
    )


def _discrete_tf_coeffs(
    sysdisc: signal.TransferFunction,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(A, B)`` discrete-time difference-equation coefficients."""
    A = np.asarray(sysdisc.den)[::-1][:-1] * (-1)
    B = np.asarray(sysdisc.num)[::-1]
    return A, B


def d2c(A, B, gamma_list, model_order: int, sample_time: float = 1.0):
    """Convert discrete-time ARX coefficients to continuous-time transfer functions."""
    ninput, norder = B.shape

    an = [1.0] + [-A[norder - 1 - j] for j in range(norder)]
    sys_tf_params_list = []

    for i in range(ninput):
        bn = [B[i, norder - 1 - j] for j in range(norder)]
        time_delay = gamma_list[i] * sample_time

        if model_order == 0:
            nums = np.array(bn) / sample_time
            dens = np.array([1, 0])
            sys_tf_params_list.append({
                "num": nums, "den": dens, "Kp": nums[-1],
                "thetap": time_delay, "BIBO_stable": False, "Method": "ZOH",
            })
            continue

        sysdisc = signal.TransferFunction(bn, an, dt=sample_time)
        real_poles = np.real(np.round(sysdisc.poles, 6))

        if np.any(real_poles < 0.0):
            # Use Tustin approximation when ZOH would place a pole in the LHP
            nums, dens = tustin_approx_d2c(bn, an, sample_time)
            params = {"num": nums, "den": dens, "thetap": time_delay,
                      "Method": "Tustin"}
            if np.all(real_poles < 1.0):
                params["Kp"] = np.sum(bn) / np.sum(an)
                params["BIBO_stable"] = True
            else:
                params["Kp"] = nums[-1]
                params["BIBO_stable"] = False
            sys_tf_params_list.append(params)
            continue

        # Stable ZOH discretisation: recover continuous model via matrix log
        ssdisc = sysdisc.to_ss()
        Ac = logm(ssdisc.A) / sample_time
        Bc = inv(inv(Ac) @ (ssdisc.A - np.eye(norder))) @ ssdisc.B
        syscont = signal.ss2tf(Ac, Bc, ssdisc.C, ssdisc.D)
        nums = np.copy(syscont[0][0])
        dens = np.copy(syscont[1])

        if np.all(real_poles < 1.0):
            Kp = np.sum(bn) / np.sum(an)
            factor = dens[-1]
            nums = nums / factor
            dens = dens / factor

            if norder == 1:
                time_constant = dens[0]
                damping_coeff = 0.0
                time_zero = 0.0
            else:
                time_constant = dens[0] ** 0.5
                damping_coeff = dens[1] / (2 * time_constant)
                nums = nums / Kp
                time_zero = nums[-2]

            sys_tf_params_list.append({
                "num": np.copy(syscont[0][0]), "den": np.copy(syscont[1]),
                "Kp": Kp, "taup": time_constant, "zetap": damping_coeff,
                "thetap": time_delay, "tauz": time_zero,
                "BIBO_stable": True, "Method": "ZOH",
            })
        else:
            sys_tf_params_list.append({
                "num": np.copy(syscont[0][0]), "den": np.copy(syscont[1]),
                "Kp": syscont[0][0][-1], "thetap": time_delay,
                "BIBO_stable": False, "Method": "ZOH",
            })

    return sys_tf_params_list


def tustin_approx_d2c(bn, an, Ts):
    """Tustin (bilinear) approximation of a discrete transfer function."""
    s = np.poly1d([1, 0])
    f = -(s * Ts + 2)
    g = (s * Ts - 2)

    bn = [0] * (len(an) - len(bn)) + list(bn)
    dendeg = len(an) - 1

    nums = 0
    dens = 0
    for i in range(dendeg + 1):
        nums += float(bn[::-1][i]) * (f ** i) * (g ** (dendeg - i))
        dens += float(an[::-1][i]) * (f ** i) * (g ** (dendeg - i))

    mfactor = dens.c[-1]
    return nums.c / mfactor, dens.c / mfactor