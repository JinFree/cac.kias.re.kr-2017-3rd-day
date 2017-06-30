#!/usr/bin/env python
import numpy as np
import distance_fortran
import time

def main():
    c1_size = 1000
    c2_size = 500

    c1 = np.random.random((c1_size,3))
    c2 = np.random.random((c2_size,3))

    repeat = 20

    print "F2py"
    t = time.time()
    for i in xrange(repeat):
        dist = distance_fortran.calc_distance(c1,c2)
    print "    Time:", time.time()-t
    print "    Check:", dist[10][10] == np.sqrt( np.sum( (c1[10]-c2[10])**2 ) )

if __name__ == '__main__':
    main()
