#!/usr/bin/env python

a = [ 'a', 'b', 'c', 'd', 'e' ]

for b in a:
    print b

for i in range(len(a)):
    print i, a[i]



a = { 'a': 1, 'b': 2, 'c': 3 }

for key in a:
    print key, a[key]

for key, value in a.iteritems():
    print key, value
