from __future__ import annotations

import numpy as np


def _compute_fit_metrics(
    Yraw: np.ndarray,
    Ypred: np.ndarray,
    start_index: int,
    yvalidstart: int,
    r2_uses_corr: bool = False,
):
    """Compute ``(SSEtrain, R2train, SSEval, R2val)`` for a fitted model.

    ``Ypred`` is the full prediction starting at ``start_index``.
    When ``yvalidstart > start_index`` the model is scored on both the
    training and validation regions; otherwise only training metrics are
    returned (and validation metrics mirror training metrics).
    """
    if yvalidstart > start_index:
        split = yvalidstart - start_index
        Ywork = Yraw[start_index:yvalidstart + 1]
        Yval = Yraw[yvalidstart:]
        Ypred_trn = Ypred[:split + 1]
        Ypred_val = Ypred[split:]
    else:
        Ywork = Yraw[start_index:]
        Yval = None
        Ypred_trn = Ypred
        Ypred_val = None

    def _score(y_true, y_hat):
        resid = y_true - y_hat
        sse = float(np.sum(resid ** 2))
        if r2_uses_corr:
            r2 = float(np.corrcoef(y_true, y_hat)[0, 1] ** 2)
        else:
            r2 = float(
                1 - np.sum(np.abs(resid)) / np.sum(np.abs(y_true - np.mean(y_true)))
            )
        return sse, r2

    SSEtrain, R2_train = _score(Ywork, Ypred_trn)
    if Yval is not None:
        SSEval, R2_val = _score(Yval, Ypred_val)
    else:
        SSEval, R2_val = SSEtrain, R2_train

    return SSEtrain, R2_train, SSEval, R2_val