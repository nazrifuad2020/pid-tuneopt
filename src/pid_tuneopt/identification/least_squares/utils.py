from __future__ import annotations

import numpy as np
from scipy import interpolate


def _prepend_constant(values: np.ndarray, count: int, fill: float) -> np.ndarray:
    """Return ``values`` with ``count`` copies of ``fill`` prepended."""
    if count <= 0:
        return values
    return np.concatenate((np.full(count, fill), values))


def _apply_time_delay(
    time: np.ndarray,
    y: np.ndarray,
    delay: float,
    sample_time: float,
) -> np.ndarray:
    """Shift ``y`` forward in time by ``delay`` using linear interpolation.

    Samples that fall before the delayed signal is available are zero-padded.
    """
    time_shifted = time + delay
    n_pad = np.arange(time[0], time_shifted[0], sample_time).shape[0]
    if n_pad == 0:
        return y
    interp = interpolate.interp1d(time_shifted, y)
    return np.concatenate((np.zeros(n_pad), interp(time[n_pad:])))