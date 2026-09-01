from numba import njit, prange, get_num_threads, set_num_threads, threading_layer
import numpy as np
import os
@njit(parallel=True)
def warmup():
    a = np.zeros(10000)
    for i in prange(10000):
        a[i] = i
    return a

# warmup()
print(get_num_threads())

print(os.environ.get('NUMBA_NUM_THREADS'))
print(os.environ.get('OMP_NUM_THREADS'))

print(threading_layer())

import multiprocessing
print(multiprocessing.cpu_count())

