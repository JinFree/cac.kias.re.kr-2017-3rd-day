#!/usr/bin/env python
import numpy as np
import math

def calc_distance_loop(c1,c2):
    c1_size = c1.shape[0]
    c2_size = c2.shape[0]
    distance = np.empty((c1_size,c2_size),dtype=np.double)
    for i in range(c1_size):
        for j in range(c2_size):
            distance[i][j] = math.sqrt( np.sum( (c1[i]-c2[j])**2 ) )
    return distance

def calc_distance(c1,c2):
    distance = np.sqrt( np.sum( ( c1[:,None,:] - c2[None,:,:] )**2, axis=2 ) )
    return distance
