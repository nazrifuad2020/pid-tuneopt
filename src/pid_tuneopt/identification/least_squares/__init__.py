"""Least-squares identification package."""

from .constants import (
    PREP_SUBTRACT_INITIAL,
    PREP_REMOVE_MEAN,
    PREP_RAW,
)
from .continuous import d2c, tustin_approx_d2c
from .fits import (
    linear_least_squares_fit_integrating,
    linear_least_squares_fit_first_order,
    linear_least_squares_fit_second_order,
)
from .identification import sysid_MISO
from .preprocessing import find_start_index, generate_training_data

__all__ = [
    "PREP_SUBTRACT_INITIAL",
    "PREP_REMOVE_MEAN",
    "PREP_RAW",
    "d2c",
    "tustin_approx_d2c",
    "linear_least_squares_fit_integrating",
    "linear_least_squares_fit_first_order",
    "linear_least_squares_fit_second_order",
    "sysid_MISO",
    "find_start_index",
    "generate_training_data",
]