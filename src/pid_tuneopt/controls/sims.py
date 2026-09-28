# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 21:25:21 2019

@author: nazri

Pure-Python port of pid_loop_sim_discrete.pyx
"""

import numpy as np

from .utils import find_peaks


def set_point_tracking(mo_Gp, Kc, Ti, Td, umin=-np.inf, umax=np.inf, D_on_Pv=1,
                       Ts=1.0, spmag=1.0, noiseVar=0.0, endtime=None,
                       gotoendtime=0):
    assert Kc > 0, "Kc must be positve"
    
    if endtime is None and gotoendtime:
        return [], [], -1

    if not ("A" in mo_Gp and "B" in mo_Gp):
        return

    Ap = mo_Gp["A"]
    Bp = mo_Gp["B"]

    gamma_p = int(np.floor(mo_Gp["thetap"] / Ts))

    # initialization
    ny = Ap.shape[0]
    nu = Bp.shape[0]

    ybefore = np.zeros(ny)
    u_before = np.zeros(ny + gamma_p)

    yholder = np.zeros(2)
    uholder = np.zeros(2)

    error = spmag
    sumerror = error
    error_prev = 0.0

    Kp = mo_Gp["Kp"]
    gain_factor = 1
    if Kp < 0:
        gain_factor = -1

    # First control move
    if D_on_Pv:
        uholder[-1] = gain_factor * Kc * (error + Ts / Ti * sumerror)
    else:
        uholder[-1] = gain_factor * Kc * (error + Ts / Ti * sumerror + Td / Ts * (error - error_prev))
    if uholder[-1] > umax:
        uholder[-1] = umax
    elif uholder[-1] < umin:
        uholder[-1] = umin
    u_before[-1] = uholder[-1]

    yprev = 0.0
    yval = 0.0

    # --- Dead-time pre-roll ------------------------------------------------
    if gamma_p > 0:
        nsize = gamma_p + 1
        y = np.zeros(nsize)
        u = np.zeros(nsize)
        u[0] = uholder[-1]
        for i in range(1, nsize):
            yval = 0.0
            for j in range(ny):
                yval += (Ap[j] * ybefore[j] + Bp[j] * u_before[j])
            
            error_prev = error
            error = spmag - yval
            sumerror += error

            if D_on_Pv:
                u[i] = gain_factor * Kc * (error + Ts / Ti * sumerror - Td / Ts * (yval - yprev))
            else:
                u[i] = gain_factor * Kc * (error + Ts / Ti * sumerror + Td / Ts * (error - error_prev))

            if u[i] > umax:
                u[i] = umax
            elif u[i] < umin:
                u[i] = umin

            y[i] = yval
            yprev = y[i]

            for j in range(ny - 1):
                ybefore[j] = ybefore[j + 1]
            ybefore[-1] = yval

            for j in range(ny + gamma_p - 1):
                u_before[j] = u_before[j + 1]
            u_before[-1] = u[i]

        yholder = np.append(yholder, y[1:])
        uholder = np.append(uholder, u[1:])

    Main_Loop_Flag = 1
    peak_loc_list = []
    first_peak = first_valley = second_peak = second_valley = 0.0

    derror = 0.0

    maxsize = 0
    if endtime is not None:
        maxsize = int(endtime / Ts) + 1
    trial = 0
    status = 0

    # Block simulation
    ynew = np.zeros(10)
    unew = np.zeros(10)
    while Main_Loop_Flag:
        unew[0] = uholder[-1]
        for iloop in range(1, 10):  # while iloop < 9:
            yval = 0.0
            for j in range(ny):
                yval += (Ap[j] * ybefore[j] + Bp[j] * u_before[j])

            error_prev = error
            error = spmag - yval
            sumerror += error
            derror = error - error_prev

            if D_on_Pv:
                unew[iloop] = gain_factor * Kc * (error + Ts / Ti * sumerror - Td / Ts * (yval - yprev))
            else:
                unew[iloop] = gain_factor * Kc * (error + Ts / Ti * sumerror + Td / Ts * derror)

            if unew[iloop] > umax:
                unew[iloop] = umax
            elif unew[iloop] < umin:
                unew[iloop] = umin

            ynew[iloop] = yval
            yprev = ynew[iloop]

            for j in range(ny - 1):
                ybefore[j] = ybefore[j + 1]
            ybefore[-1] = yval

            for j in range(ny + gamma_p - 1):
                u_before[j] = u_before[j + 1]
            u_before[-1] = unew[iloop]

        yholder = np.append(yholder, ynew[1:])
        uholder = np.append(uholder, unew[1:])

        if not gotoendtime:
            if abs(error) < 0.01 * abs(spmag) and abs(derror / Ts) < 0.0001:
                trial += 1
                if trial == 2:
                    # normal exit
                    status = 2
                    break
            else:
                trial = 0

        if Main_Loop_Flag:
            find_peaks(yholder, peak_loc_list)
            if len(peak_loc_list) == 4:
                first_peak = yholder[peak_loc_list[0]]
                first_valley = yholder[peak_loc_list[1]]
                second_peak = yholder[peak_loc_list[2]]
                second_valley = yholder[peak_loc_list[3]]
                if abs(second_peak - second_valley) - abs(first_peak - first_valley) > 1.0:
                    # controller is not stable, force exit from the main loop
                    status = 0
                    break

        if maxsize > 0:
            if yholder.shape[0] >= maxsize:
                Main_Loop_Flag = 0
                status = 1
                break

    return uholder, yholder, status


def disturbance_rejection(mo_Gp, mo_Gd, Kc, Ti, Td, Ts=1.0, idistmag=1.0,
                          noiseVar=0.0, endtime=None, gotoendtime=0):
    assert Kc > 0, "Kc must be positve"
    
    if endtime is None and gotoendtime:
        return [], [], -1

    # --- Plant model -------------------------------------------------------
    if not ("A" in mo_Gp and "B" in mo_Gp):
        return

    Ap = mo_Gp["A"]
    Bp = mo_Gp["B"]

    gamma_p = int(np.floor(mo_Gp["thetap"] / Ts))

    # --- Disturbance model -------------------------------------------------
    if not ("A" in mo_Gd and "B" in mo_Gd):
        return

    Ad = mo_Gd["A"]
    Bd = mo_Gd["B"]

    yholder = np.zeros(1)
    uholder = np.zeros(1)

    # initialization
    ny = Ap.shape[0]

    ybefore = np.zeros(ny)
    u_before = np.zeros(ny + gamma_p)

    nydist = Ad.shape[0]

    yd_before = np.zeros(nydist)
    ud_before = np.zeros(nydist)

    error = 0.0
    sumerror = 0.0
    error_prev = 0.0
    derror = 0.0
    first_peak = first_valley = second_peak = second_valley = 0.0

    Kp = mo_Gp["Kp"]
    gain_factor = 1
    if Kp < 0:
        gain_factor = -1

    yvald = 0.0
    yval = 0.0
    yval_all = 0.0

    Main_Loop_Flag = 1
    peak_loc_list = []
    maxsize = 0
    if endtime is not None:
        maxsize = int(endtime / Ts) + 1

    trial = 0
    ymax = 0.0
    status = 0

    yall = np.zeros(10)
    unew = np.zeros(10)
    while Main_Loop_Flag:
        unew[0] = uholder[-1]
        iloop = 0
        for iloop in range(1, 10):
            yvald = 0.0
            for j in range(nydist):
                yvald += (Ad[j] * yd_before[j] + Bd[j] * ud_before[j])

            yval = 0.0
            for j in range(ny):
                yval += (Ap[j] * ybefore[j] + Bp[j] * u_before[j])

            yval_all = yvald + yval

            # if counter % control_p == 0:
            error_prev = error
            error = -yval_all
            sumerror += error
            derror = error - error_prev

            unew[iloop] = gain_factor * Kc * (error + Ts / Ti * sumerror + Td / Ts * derror)

            yall[iloop] = yval_all

            for j in range(ny - 1):
                ybefore[j] = ybefore[j + 1]
            ybefore[-1] = yval

            for j in range(nydist - 1):
                yd_before[j] = yd_before[j + 1]
            yd_before[-1] = yvald

            for j in range(ny + gamma_p - 1):
                u_before[j] = u_before[j + 1]
            u_before[-1] = unew[iloop]

            for j in range(nydist - 1):
                ud_before[j] = ud_before[j + 1]
            ud_before[-1] = idistmag

        yholder = np.append(yholder, yall[1:])
        uholder = np.append(uholder, unew[1:])

        if not gotoendtime:
            ymax = np.max(np.abs(yholder))
            if abs(error) < 0.01 * ymax and abs(derror / Ts) < 0.0001:
                trial += 1
                if trial == 2:
                    status = 2
                    break
            else:
                trial = 0

        if Main_Loop_Flag:
            find_peaks(yholder, peak_loc_list)
            if len(peak_loc_list) == 4:
                first_peak = yholder[peak_loc_list[0]]
                first_valley = yholder[peak_loc_list[1]]
                second_peak = yholder[peak_loc_list[2]]
                second_valley = yholder[peak_loc_list[3]]
                if abs(second_peak - second_valley) - abs(first_peak - first_valley) > 1.0:
                    # controller is not stable, force exit from the main loop
                    status = 0
                    break

        if maxsize > 0:
            if yholder.shape[0] >= maxsize:
                Main_Loop_Flag = 0
                status = 1
                break

    return uholder, yholder, status
