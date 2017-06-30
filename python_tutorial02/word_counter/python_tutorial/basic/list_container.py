#!/usr/bin/env python

######
# list
######
a = range(10)
print a
a.insert(1,20)
print a
b = a.pop(4)
print a
print b
c = a.pop()
print a
print c

# slicing
print a[0:5:2]

# merging
d = a + [ 'a', 'b', 'c' ]
print d

#######
# tuple
#######

a = (1,3,2,4)
print a[2]
a[2] = 7 # Error

