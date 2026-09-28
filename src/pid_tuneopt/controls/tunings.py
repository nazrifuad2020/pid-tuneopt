# -*- coding: utf-8 -*-
"""
PID tuning.

@author: nazri
"""

# ===========================================================================
# IMC tuning
# ===========================================================================
def imc_tuner(mo_Gp, tauc, pid_type=0, Ts=1.0):
    """Inner-model-control tuning rules for FOPDT / SOPDT processes."""
    Kp    = mo_Gp["Kp"]
    theta = mo_Gp["thetap"] + Ts / 2

    if "taup" not in mo_Gp:
        # First-order plant
        if pid_type == 1:
            Kc = (2 * tauc + theta) / (Kp * (tauc + theta) ** 2)
            Ti = 2 * tauc + theta
            Td = 0.0
        else:
            Kc = (2 * tauc + theta) / (Kp * (tauc + theta / 2) ** 2)
            Ti = 2 * tauc + theta
            Td = (tauc * theta + theta ** 2 / 4) / (2 * tauc + theta)
        return Kc, Ti, Td

    # Second-order plant
    taup, zetap, tauz = mo_Gp["taup"], mo_Gp["zetap"], mo_Gp["tauz"]

    if zetap == 0:
        if pid_type == 1:
            Kc = taup / (Kp * (tauc + taup))
            Ti = taup
            Td = 0.0
        else:
            Kc = (taup + theta / 2) / (Kp * (tauc + theta / 2))
            Ti = taup + theta / 2
            Td = taup * theta / (2 * taup + theta)
    elif tauz == 0.0:
        Kc = 2 * zetap * taup / (Kp * (tauc + theta))
        Ti = 2 * zetap * taup
        Td = taup / (2 * zetap)
    elif tauz > 0.0:
        a  = 2 * zetap * taup - tauz
        Kc = a / (Kp * (tauc + theta))
        Ti = a
        Td = (taup ** 2 - a * tauz) / a
    else:
        denom = tauc + tauz + theta
        tmp   = tauz * theta / denom
        a     = 2 * zetap * taup + tmp
        Kc = a / (Kp * denom)
        Ti = a
        Td = tmp + taup ** 2 / a

    if Kp < 0:
        Kc *= -1

    return Kc, Ti, Td