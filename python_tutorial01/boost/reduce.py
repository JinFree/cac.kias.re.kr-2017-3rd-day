#!/usr/bin/env python
import time

data = range(1,10000)

print 'REDUCE'
t = time.time()
product = reduce(lambda x, y: x*y, data)
print '    Time', time.time()-t

print 'LOOP'
t = time.time()
product = data[0]
prod = lambda x, y: x*y
for i in data[1:]:
    product = prod(product,i)
print '    Time', time.time()-t
