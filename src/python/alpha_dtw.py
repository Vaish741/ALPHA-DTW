
"""
Alpha-DTW
------------------------------------------------------------------

This module provides an implementation of Alpha-DTW, an extension of
classical Dynamic Time Warping (DTW) that incorporates an additional
penalty term to reduce triangle inequality violations.

Author: Vaishnavi Rastogi
"""

from __future__ import annotations
from typing import Sequence
import numpy as np


def _to_1d_array(x: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Convert input to a contiguous 1D NumPy array of type float64.

    Parameters
    ----------
    x : array-like
        Input time series.

    Returns
    -------
    np.ndarray
        1D NumPy array.

    Raises
    ------
    ValueError
        If input is not 1D or is empty.
    """
    arr = np.asarray(x, dtype=np.float64)

    if arr.ndim != 1:
        raise ValueError("Input time series must be 1-dimensional.")
    if arr.size == 0:
        raise ValueError("Input time series must not be empty.")

    return np.ascontiguousarray(arr)


def compute_alpha_penalty_matrix(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Compute the Alpha-DTW penalty matrix.

    The penalty is defined as:
        P(i,j) = alpha * [ (x[i] - x[j])^2 + (y[i] - y[j])^2 ]

    This captures intra-series structural deviations.

    Parameters
    ----------
    x, y : array-like of shape (n,)
        Input time series.

    alpha : float
        Penalty parameter. If alpha = 0, the penalty matrix is zero.

    Returns
    -------
    np.ndarray of shape (n, m)
        Penalty matrix.

    Raises
    ------
    ValueError
        If alpha ≠ 0 and lengths of x and y differ.
        
    Note
    ------
    For alpha ≠ 0, both time series must have equal length due to the
    definition of the penalty matrix.    
    """
    x_arr = _to_1d_array(x)
    y_arr = _to_1d_array(y)

    n, m = len(x_arr), len(y_arr)

    penalty = np.zeros((n, m), dtype=np.float64)

    if alpha == 0:
        return penalty

    if n != m:
        raise ValueError(
            "For alpha ≠ 0, both time series must have equal length."
        )

    for i in range(n):
        for j in range(m):
            dx = x_arr[i] - x_arr[j]
            dy = y_arr[i] - y_arr[j]
            penalty[i, j] = alpha * (dx * dx + dy * dy)

    return penalty


def alpha_dtw_distance(
    s1: Sequence[float] | np.ndarray,
    s2: Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.5
) -> float:
    """
    Compute the Alpha-DTW distance between two time series.

    This method extends classical DTW by incorporating a penalty term.
    

    The accumulated cost matrix D is defined as:

        D(i,j) = d(i,j) + min{
            D(i-1,j),     # insertion
            D(i,j-1),     # deletion
            D(i-1,j-1)    # match
        }

    where:

        d(i,j) = (s1[i] - s2[j])^2 + P(i,j)

    Parameters
    ----------
    s1, s2 : array-like of shape (n,), (m,)
        Input time series.

    alpha : float, default=0.5
        Penalty strength.
        - alpha = 0 → reduces to classical DTW
        - higher alpha → discourages non-diagonal alignments

    Returns
    -------
    float
        Alpha-DTW distance.

    Notes
    -----
    
    - Time complexity: O(n * m)
    """

    s1_arr = _to_1d_array(s1)
    s2_arr = _to_1d_array(s2)

    r, c = len(s1_arr), len(s2_arr)

    penalty = compute_alpha_penalty_matrix(s1_arr, s2_arr, alpha)

    # Initialize accumulated cost matrix
    D = np.full((r, c), np.inf, dtype=np.float64)

    # Base case
    D[0, 0] = (s1_arr[0] - s2_arr[0]) ** 2 + penalty[0, 0]

    # First column
    for i in range(1, r):
        cost = (s1_arr[i] - s2_arr[0]) ** 2 + penalty[i, 0]
        D[i, 0] = cost + D[i - 1, 0]

    # First row
    for j in range(1, c):
        cost = (s1_arr[0] - s2_arr[j]) ** 2 + penalty[0, j]
        D[0, j] = cost + D[0, j - 1]

    # Main dynamic programming loop
    for i in range(1, r):
        for j in range(1, c):
            cost = (s1_arr[i] - s2_arr[j]) ** 2 + penalty[i, j]

            D[i, j] = cost + min(
                D[i - 1, j],
                D[i, j - 1],
                D[i - 1, j - 1]
            )

    return float(np.sqrt(D[r - 1, c - 1]))


if __name__ == "__main__":
    # Simple sanity check
    x = [1, 1,1,-1,1,1,1]
    y = [1,1,-1,1,1,1,1]

    dist = alpha_dtw_distance(x, y, alpha=0.5)
    print(f"Alpha-DTW distance: {dist:.4f}")
    dist = alpha_dtw_distance(x, y, alpha=0)
    print(f"DTW distance: {dist:.4f}")