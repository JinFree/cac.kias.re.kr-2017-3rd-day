#!/usr/bin/env python
import dsum
import cydsum
import time
import numpy as np

data = np.arange(10000.0)

print 'dsum(data)', dsum.dsum(data), cydsum.dsum(data)

repeat = 1000
t = time.time()
for i in range(repeat):
    dsum.dsum(data)
print '    Python Time', time.time()-t

t = time.time()
for i in range(repeat):
    cydsum.dsum(data)
print '    Cython Time', time.time()-t

t = time.time()
for i in range(repeat):
    np.sum(data)
print '    Numpy Time', time.time()-t
