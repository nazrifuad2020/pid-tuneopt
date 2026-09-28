import numpy as np

from scipy import signal, interpolate


def find_peaks(yresp, peak_loc_list):
    """
    Locate up to the last 4 turning points (peaks/valleys) in `yresp`.

    `peak_loc_list` is a mutable Python list used both as input (state from
    previous call) and output.  It always holds at most 4 elements.
    """
    if len(peak_loc_list) == 0:
        start_index_arr = np.argwhere(np.abs(yresp) > 0.0)
        if start_index_arr.shape[0] > 0:
            start_ind = start_index_arr[0][0]
        else:
            return
        if start_ind == yresp.shape[0] - 1:
            return
    else:
        start_ind = peak_loc_list[-1]

    if yresp[start_ind + 1] > yresp[start_ind]:
        switch = 1
    else:
        switch = 0

    nsize = yresp.shape[0] - 1
    while start_ind < nsize:
        start_ind += 1
        if switch == 1:
            if yresp[start_ind] < yresp[start_ind - 1]:
                if len(peak_loc_list) < 4:
                    peak_loc_list.append(start_ind - 1)
                else:
                    peak_loc_list.pop(0)
                    peak_loc_list.append(start_ind - 1)
                switch = 0
        else:
            if yresp[start_ind] > yresp[start_ind - 1]:
                if len(peak_loc_list) < 4:
                    peak_loc_list.append(start_ind - 1)
                else:
                    peak_loc_list.pop(0)
                    peak_loc_list.append(start_ind - 1)
                switch = 1
    return


def find_settling_time(tresp, yresp, spmag=1.0):
    yohoo = np.abs(spmag - yresp) / abs(spmag)
    yahaa = yohoo[::-1]
    tahaa = tresp[::-1]
    x_loc = np.where(yahaa > 0.05)
    if x_loc[0].size > 0:
        Tsettle_SP = (tahaa[x_loc[0][0]] - tahaa[x_loc[0][0] - 1]) / \
                     (yahaa[x_loc[0][0]] - yahaa[x_loc[0][0] - 1]) * \
                     (0.05 - yahaa[x_loc[0][0] - 1]) + tahaa[x_loc[0][0] - 1]
    else:
        Tsettle_SP = float("inf")

    return Tsettle_SP


def stability_analysis(Kc, Ti, Td, Gpmodel, Ts=1.0):
    Kp = Gpmodel["Kp"]
    if Kp < 0:
        Kc *= -1
    p_numGc = np.poly1d([Kc * Td, Kc, Kc / Ti])
    p_denGc = np.poly1d([1, 0])
    p_numGp = np.poly1d(Gpmodel["num"])
    p_denGp = np.poly1d(Gpmodel["den"])
    p_numGOL = p_numGc * p_numGp
    p_denGOL = p_denGc * p_denGp
    GOL = signal.TransferFunction(p_numGOL, p_denGOL)
    win = np.logspace(-2, 2)
    w, mag, phase = signal.bode(GOL, win)
    AR = 10 ** (mag / 20)
    theta = Gpmodel["thetap"] + Ts / 2
    phase -= (180 / np.pi) * theta * w
    finterp = interpolate.interp1d(phase, w, fill_value="extrapolate")

    # --- gain margin: extract a genuine Python float from the array ---
    wc = float(np.atleast_1d(finterp(-180.0)).ravel()[0])
    _, mag_c, _ = signal.bode(GOL, np.atleast_1d(wc))
    ARc = float(10 ** (np.atleast_1d(mag_c).ravel()[0] / 20))
    GM = 1.0 / ARc

    # --- phase margin: same treatment ---
    finterp = interpolate.interp1d(AR, w, fill_value="extrapolate")
    wg = float(np.atleast_1d(finterp(1.0)).ravel()[0])
    _, _, phase_g = signal.bode(GOL, np.atleast_1d(wg))
    phase_g = float(np.atleast_1d(phase_g).ravel()[0]) - (180 / np.pi) * theta * wg
    PM = phase_g + 180
    max_delay = PM / wg * (np.pi / 180)

    return float(GM), float(PM), float(max_delay)


def find_Kcmax(Gpmodel, Ts=1.0):
    num = Gpmodel["num"]
    den = Gpmodel["den"]
    theta = Gpmodel["thetap"] + Ts / 2
    
    syscont = signal.TransferFunction(num, den)
    
    win = np.logspace(-2, 2)
    
    w, mag, phase = signal.bode(syscont, win)
    
    if theta > 0.0:
        phase -= (180 / np.pi) * theta * w
    
    phase_min = np.min(phase)
    if phase_min < -180.0:
        # find critical freq.
        finterp = interpolate.interp1d(phase, w, fill_value="extrapolate")
        wc = finterp(-180.0)
        wg, mag_g, phase_c = signal.bode(syscont, wc)
    
        # convert from dB to amplitude ratio
        mag_g = 10 ** (mag_g / 20)
        Kcmax = float(1 / mag_g[0])
    else:
        Kp = np.abs(Gpmodel["Kp"])
        Kcmax = 10 / Kp
        i = 0
        while True:
            if 10 ** i > Kcmax:
                Kcmax = 10 ** i
                break
            i += 1

    return Kcmax


def find_peaks_wrapper(yresp):
    peak_loc_list = []
    find_peaks(yresp, peak_loc_list)
    return list(peak_loc_list)