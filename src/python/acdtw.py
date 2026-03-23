

"""Python implementation of ACDTW for equal length time series.



Notes
-----
- This implementation is defined only for equal-length one-dimensional series.
- The local cost is ``(Q[i] - C[j])**2``.
- Penalties are applied only to reuse moves (up/left); diagonal moves do not
  receive the extra reuse penalty.
- 
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def _as_1d_float_array(x: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("Each time series must be 1D.")
    if arr.size == 0:
        raise ValueError("Each time series must be non-empty.")
    return np.ascontiguousarray(arr, dtype=np.float64)


def acdtw_equal_len_paper(Q: Sequence[float] | np.ndarray, C: Sequence[float] | np.ndarray) -> float:
    """Compute the equal-length ACDTW distance.

    Parameters
    ----------
    Q, C : array-like of shape (n,)
        Input time series. They must have the same length.

    Returns
    -------
    float
        Final cumulative ACDTW cost.
    """
    q_arr = _as_1d_float_array(Q)
    c_arr = _as_1d_float_array(C)

    m = int(q_arr.shape[0])
    n = int(c_arr.shape[0])
    if m != n:
        raise ValueError("Equal-length only (m == n).")

    D = np.empty((m, n), dtype=np.float64)
    qcnt = np.zeros((m, n), dtype=np.int32)
    ccnt = np.zeros((m, n), dtype=np.int32)

    diff = float(q_arr[0] - c_arr[0])
    D[0, 0] = diff * diff
    qcnt[0, 0] = 1
    ccnt[0, 0] = 1

    for i in range(1, m):
        diff = float(q_arr[i] - c_arr[0])
        dij = diff * diff
        D[i, 0] = D[i - 1, 0] + dij + (ccnt[i - 1, 0] * dij)
        qcnt[i, 0] = 1
        ccnt[i, 0] = ccnt[i - 1, 0] + 1

    for j in range(1, n):
        diff = float(q_arr[0] - c_arr[j])
        dij = diff * diff
        D[0, j] = D[0, j - 1] + dij + (qcnt[0, j - 1] * dij)
        qcnt[0, j] = qcnt[0, j - 1] + 1
        ccnt[0, j] = 1

    for i in range(1, m):
        for j in range(1, n):
            diff = float(q_arr[i] - c_arr[j])
            dij = diff * diff

            dup = D[i - 1, j] + dij + (ccnt[i - 1, j] * dij)
            ddiag = D[i - 1, j - 1] + dij
            dleft = D[i, j - 1] + dij + (qcnt[i, j - 1] * dij)

            minv = dup
            argmin_idx = 0
            if ddiag < minv:
                minv = ddiag
                argmin_idx = 1
            if dleft < minv:
                minv = dleft
                argmin_idx = 2

            if argmin_idx == 0:
                D[i, j] = dup
                qcnt[i, j] = 1
                ccnt[i, j] = ccnt[i - 1, j] + 1
            elif argmin_idx == 1:
                D[i, j] = ddiag
                qcnt[i, j] = 1
                ccnt[i, j] = 1
            else:
                D[i, j] = dleft
                qcnt[i, j] = qcnt[i, j - 1] + 1
                ccnt[i, j] = 1

    return np.sqrt(D[m - 1, n - 1])
