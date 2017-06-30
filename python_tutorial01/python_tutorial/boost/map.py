#!/usr/bin/env python
import time

data = range(10000)

print 'MAP'
t = time.time() 
new_data = map(float,data)
print '    Time', time.time() - t

print 'LOOP'
t = time.time() 
new_data = []
for i in data:
    new_data.append(float(i))
print '    Time', time.time() - t
