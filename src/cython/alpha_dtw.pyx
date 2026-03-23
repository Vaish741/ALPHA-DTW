# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:46:47 2026

@author: vaish
"""

# alpha_dtw.pyx

import numpy as np
cimport numpy as np

from libc.math cimport sqrt
from cython cimport boundscheck, wraparound


@boundscheck(False)
@wraparound(False)
cpdef double alpha_dtw_distance(
    object s1,
    object s2,
    double alpha=0.5
):
    """
    Cython implementation of Alpha-DTW distance.

    
    ----------
    s1, s2 : array-like
        Input time series (1D)

    alpha : float
        Penalty parameter

    Returns
    -------
    float
        Alpha-DTW distance
    """

    
    cdef np.ndarray[np.float64_t, ndim=1] a
    cdef np.ndarray[np.float64_t, ndim=1] b

    a = np.ascontiguousarray(s1, dtype=np.float64)
    b = np.ascontiguousarray(s2, dtype=np.float64)

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Input time series must be 1-dimensional.")

    if a.size == 0 or b.size == 0:
        raise ValueError("Input time series must not be empty.")

    cdef int i, j
    cdef int n = a.shape[0]
    cdef int m = b.shape[0]

    cdef np.ndarray[np.float64_t, ndim=2] penalty = np.zeros((n, m), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] D = np.full((n, m), np.inf, dtype=np.float64)

    cdef double dx, dy, cost

    
    if alpha != 0:
        if n != m:
            raise ValueError("For alpha ≠ 0, both time series must have equal length.")

        for i in range(n):
            for j in range(m):
                dx = a[i] - a[j]
                dy = b[i] - b[j]
                penalty[i, j] = alpha * (dx * dx + dy * dy)

    # --- Step 1: Initialize DP matrix ---
    D[0, 0] = (a[0] - b[0]) * (a[0] - b[0]) + penalty[0, 0]

    # First column
    for i in range(1, n):
        cost = (a[i] - b[0]) * (a[i] - b[0]) + penalty[i, 0]
        D[i, 0] = cost + D[i - 1, 0]

    # First row
    for j in range(1, m):
        cost = (a[0] - b[j]) * (a[0] - b[j]) + penalty[0, j]
        D[0, j] = cost + D[0, j - 1]

    # --- Step 2: Main DP loop ---
    for i in range(1, n):
        for j in range(1, m):
            cost = (a[i] - b[j]) * (a[i] - b[j]) + penalty[i, j]

            D[i, j] = cost + min(
                D[i - 1, j],
                D[i, j - 1],
                D[i - 1, j - 1]
            )

    # --- Step 3: Return result ---
    return sqrt(D[n - 1, m - 1])