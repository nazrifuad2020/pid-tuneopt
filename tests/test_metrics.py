import numpy as np

from pid_tuneopt.identification.least_squares.metrics import _compute_fit_metrics


def test_perfect_prediction_returns_zero_sse_and_r2_one():
    Yraw = np.arange(10, dtype=float)
    start_index = 2
    yvalidstart = 5
    Ypred = Yraw[start_index:]
    sse_trn, r2_trn, sse_val, r2_val = _compute_fit_metrics(
        Yraw, Ypred, start_index, yvalidstart
    )
    assert sse_trn == 0.0
    assert sse_val == 0.0
    assert np.isclose(r2_trn, 1.0)
    assert np.isclose(r2_val, 1.0)


def test_no_validation_split_validation_mirrors_training():
    Yraw = np.arange(10, dtype=float)
    Ypred = Yraw.copy()
    sse_trn, r2_trn, sse_val, r2_val = _compute_fit_metrics(
        Yraw, Ypred, start_index=0, yvalidstart=0
    )
    assert sse_trn == sse_val
    assert r2_trn == r2_val


def test_constant_prediction_yields_r2_zero():
    Yraw = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    Ypred = np.full_like(Yraw, np.mean(Yraw))
    _, r2_trn, _, _ = _compute_fit_metrics(
        Yraw, Ypred, start_index=0, yvalidstart=0
    )
    assert np.isclose(r2_trn, 0.0)


def test_r2_uses_corr_flag():
    Yraw = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    Ypred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    _, r2_std, _, _ = _compute_fit_metrics(Yraw, Ypred, 0, 0, r2_uses_corr=False)
    _, r2_cor, _, _ = _compute_fit_metrics(Yraw, Ypred, 0, 0, r2_uses_corr=True)
    # Both metrics should be high for a linear-shifted prediction
    assert r2_std > 0.9
    assert np.isclose(r2_cor, 1.0, atol=1e-10)


def test_training_and_validation_regions_differ():
    Yraw = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    # Perfect training region, wrong validation region
    Ypred = Yraw.copy()
    Ypred[5:] += 100.0
    sse_trn, _, sse_val, _ = _compute_fit_metrics(
        Yraw, Ypred, start_index=0, yvalidstart=4
    )
    assert sse_trn == 0.0
    assert sse_val > 0.0