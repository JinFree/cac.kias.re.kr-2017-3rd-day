import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "distance.h" nogil:
    void calc_distance(int cm1, int cm2, double** c1, double** c2, double** distance)

def calc_distance_wrap(np.ndarray [np.double_t,ndim=2,mode='c'] c1, np.ndarray [np.double_t,ndim=2] c2):
    cdef int c1_size = c1.shape[0] 
    cdef int c2_size = c2.shape[0] 
    cdef np.ndarray [np.double_t,ndim=2] distance = np.empty((c1_size,c2_size),dtype=np.double)

    cdef double** c1_arr
    cdef double** c2_arr
    cdef double** distance_arr

    cdef int i

    # memory allocation for 2D array
    c1_arr = <double**> malloc( sizeof(double*) * c1_size )
    c2_arr = <double**> malloc( sizeof(double*) * c2_size )
    distance_arr = <double**> malloc( sizeof(double*) * c1_size )

    for i in range(c1_size):
        c1_arr[i] = &c1[i,0] 
    for i in range(c2_size):
        c2_arr[i] = &c2[i,0]
    for i in range(c1_size):
        distance_arr[i] = &distance[i,0]

    calc_distance(c1_size,c2_size,c1_arr,c2_arr,distance_arr)

    # free memory
    free(c1_arr)
    free(c2_arr)
    free(distance_arr)

    return distance
