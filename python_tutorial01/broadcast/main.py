#!/usr/bin/env python
import numpy as np

a = np.arange(15).reshape((5,3))
print a
print a.shape

b = np.array([0.1, 0.2, 0.3])
print b
print b.shape

c = a+b
print c
print c.shape
