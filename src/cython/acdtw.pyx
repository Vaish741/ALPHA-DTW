# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 16:09:03 2026

@author: vaish
"""

# acdtw.pyx

import numpy as np
cimport numpy as np

from libc.math cimport sqrt
from cython cimport boundscheck, wraparound


@boundscheck(False)
@wraparound(False)
cpdef double acdtw_equal_len_paper(object Q, object C):
    """
    Cython implementation of equal-length ACDTW.

    
    """

    # --- Input conversion (same as Python) ---
    cdef np.ndarray[np.float64_t, ndim=1] q_arr
    cdef np.ndarray[np.float64_t, ndim=1] c_arr

    q_arr = np.ascontiguousarray(Q, dtype=np.float64)
    c_arr = np.ascontiguousarray(C, dtype=np.float64)

    if q_arr.ndim != 1 or c_arr.ndim != 1:
        raise ValueError("Each time series must be 1D.")

    if q_arr.size == 0 or c_arr.size == 0:
        raise ValueError("Each time series must be non-empty.")

    cdef int m = q_arr.shape[0]
    cdef int n = c_arr.shape[0]

    if m != n:
        raise ValueError("Equal-length only (m == n).")

    # --- Allocate matrices ---
    cdef np.ndarray[np.float64_t, ndim=2] D = np.empty((m, n), dtype=np.float64)
    cdef np.ndarray[np.int32_t, ndim=2] qcnt = np.zeros((m, n), dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=2] ccnt = np.zeros((m, n), dtype=np.int32)

    cdef int i, j, argmin_idx
    cdef double diff, dij, dup, ddiag, dleft, minv

    # --- Base case ---
    diff = q_arr[0] - c_arr[0]
    D[0, 0] = diff * diff
    qcnt[0, 0] = 1
    ccnt[0, 0] = 1

    # --- First column ---
    for i in range(1, m):
        diff = q_arr[i] - c_arr[0]
        dij = diff * diff

        D[i, 0] = D[i - 1, 0] + dij + (ccnt[i - 1, 0] * dij)
        qcnt[i, 0] = 1
        ccnt[i, 0] = ccnt[i - 1, 0] + 1

    # --- First row ---
    for j in range(1, n):
        diff = q_arr[0] - c_arr[j]
        dij = diff * diff

        D[0, j] = D[0, j - 1] + dij + (qcnt[0, j - 1] * dij)
        qcnt[0, j] = qcnt[0, j - 1] + 1
        ccnt[0, j] = 1

    # --- Main DP loop ---
    for i in range(1, m):
        for j in range(1, n):
            diff = q_arr[i] - c_arr[j]
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

    return sqrt(D[m - 1, n - 1])