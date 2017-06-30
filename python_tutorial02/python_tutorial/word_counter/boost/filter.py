#!/usr/bin/env python
import time

data = range(-10000,10000)

print 'FILTER'
t = time.time()
new_data = filter(lambda x: x < 0, data)
print '    Time', time.time()-t

print 'LOOP'
t = time.time()
new_data = []
for i in data:
    if i < 0:
        new_data.append(i)
print '    Time', time.time()-t
