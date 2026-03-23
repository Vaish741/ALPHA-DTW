# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:57:34 2026

@author: vaish
"""

import numpy as np

# Import Python version
from src.python.alpha_dtw import alpha_dtw_distance as python_dtw

# Import Cython version
from src.cython.alpha_dtw import alpha_dtw_distance as cython_dtw


def run_test():
    # Example data
    x = np.array([1, 1, 1, -1, 1, 1, 1], dtype=np.float64)
    y = np.array([1, 1, -1, 1, 1, 1, 1], dtype=np.float64)

    alpha = 0.5

    d_py = python_dtw(x, y, alpha=alpha)
    
    d_cy = cy_dtw(x, y, alpha)

    print("Python Alpha-DTW:", d_py)
    print("Cython ALpha-DTW:", d_cy)

    


if __name__ == "__main__":
    run_test()