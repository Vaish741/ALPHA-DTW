import numpy as np

from src.python.alpha_dtw import alpha_dtw_distance as py_alpha
from src.cython.alpha_dtw import alpha_dtw_distance as cy_alpha

from src.python.acdtw import acdtw_equal_len_paper as py_acdtw
from src.cython.acdtw import acdtw_equal_len_paper as cy_acdtw

x = np.array([1,1,1,-1,1,1,1], dtype=np.float64)
y = np.array([1,1,-1,1,1,1,1], dtype=np.float64)

print("Alpha Python:", py_alpha(x, y, alpha=0.5))
print("Alpha Cython:", cy_alpha(x, y, 0.5))

print("ACDTW Python:", py_acdtw(x, y))
print("ACDTW Cython:", cy_acdtw(x, y))