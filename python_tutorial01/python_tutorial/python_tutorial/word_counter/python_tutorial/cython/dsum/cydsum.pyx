import numpy as np
cimport numpy as np
import cython

@cython.wraparound(False)
@cython.boundscheck(False)
def dsum(np.ndarray [np.double_t,ndim=1] data):
    cdef double ret = 0.0
    cdef int i
    cdef int data_size = data.size
    for i in range(data_size):
        ret += data[i]
    return ret
