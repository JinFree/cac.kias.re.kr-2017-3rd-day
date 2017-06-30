#!/usr/bin/env python
import fib
import cyfib1
import cyfib2
import cfib_wrap
import nbfib
import time

print 'fib(10)', fib.fib(10), cyfib1.fib(10), cyfib2.fib(10), cfib_wrap.fib(10), nbfib.fib(10)
print 'fib(50)', fib.fib(50), cyfib1.fib(50), cyfib2.fib(50), cfib_wrap.fib(50), nbfib.fib(50)
print 'fib(100)', fib.fib(100), cyfib1.fib(100), cyfib2.fib(100), cfib_wrap.fib(100), nbfib.fib(100)
print 'fib(200)', fib.fib(200), cyfib1.fib(200), cyfib2.fib(200), cfib_wrap.fib(200), nbfib.fib(200)

repeat = 10000
t = time.time()
for i in range(repeat):
    fib.fib(50)
print '    Python Time', time.time()-t

t = time.time()
for i in range(repeat):
    cyfib1.fib(50)
print '    Cython1 Time', time.time()-t

t = time.time()
for i in range(repeat):
    cyfib2.fib(50)
print '    Cython2 Time', time.time()-t

t = time.time()
for i in range(repeat):
    cfib_wrap.fib(50)
print '    CWRAP Time', time.time()-t

t = time.time()
for i in range(repeat):
    nbfib.fib(50)
print '    Numba Time', time.time()-t
