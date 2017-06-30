#cython: boundscheck=True, wraparound=True
import numpy as np
cimport numpy as np
from libc.math cimport sqrt
cimport cython

#@cython.boundscheck(False)
#@cython.wraparound(False)
def calc_distance_cython(np.ndarray [np.double_t,ndim=2,mode='c'] c1, np.ndarray [np.double_t,ndim=2] c2):
    cdef int c1_size = c1.shape[0]
    cdef int c2_size = c2.shape[0]
    cdef double dx, dy, dz
    cdef np.ndarray [np.double_t,ndim=2] distance = np.empty((c1_size,c2_size),dtype=np.double)
    cdef int i, j

    for i in range(c1_size):
        for j in range(c2_size):
            dx = c1[i,0] - c2[j,0]
            dy = c1[i,1] - c2[j,1]
            dz = c1[i,2] - c2[j,2]
            distance[i,j] = sqrt(dx*dx + dy*dy + dz*dz)

    return distance
