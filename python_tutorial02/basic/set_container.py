#!/usr/bin/env python

a = ['a',1,'b','a',7,1]
print a
b = set(a)
print b
b.remove(7)
print b
b.add(3.14)
print b
