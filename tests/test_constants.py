from pid_tuneopt.identification.least_squares import (
    PREP_SUBTRACT_INITIAL,
    PREP_REMOVE_MEAN,
    PREP_RAW,
)


def test_constants_are_distinct_and_int():
    values = [PREP_SUBTRACT_INITIAL, PREP_REMOVE_MEAN, PREP_RAW]
    assert all(isinstance(v, int) for v in values)
    assert len(set(values)) == 3


def test_constants_expected_values():
    assert PREP_SUBTRACT_INITIAL == 0
    assert PREP_REMOVE_MEAN == 1
    assert PREP_RAW == 2