#!/usr/bin/env python
import numpy as np
import distance_py
import time

def main():
    c1_size = 1000
    c2_size = 500

    c1 = np.random.random((c1_size,3))
    c2 = np.random.random((c2_size,3))

    repeat = 20

    print "Numba"
    t = time.time()
    for i in xrange(repeat):
        dist = distance_py.calc_distance_loop(c1,c2)
    print "    Time:", time.time()-t
    print "    Check:", dist[10][10] == np.sqrt( np.sum( (c1[10]-c2[10])**2 ) )

    print "Numba (typed)"
    t = time.time()
    for i in xrange(repeat):
        dist = distance_py.calc_distance_loop2(c1,c2)
    print "    Time:", time.time()-t
    print "    Check:", dist[10][10] == np.sqrt( np.sum( (c1[10]-c2[10])**2 ) )

if __name__ == '__main__':
    main()
