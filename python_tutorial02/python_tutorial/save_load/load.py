#!/usr/bin/env python
import cPickle as pkl
import numpy as np

# numpy text format
data = np.loadtxt('data.txt')
print 'data.txt'
print data
# numpy binary
data = np.load('data.npy')
print 'data.npy'
print data
# pickle
with open('data.pkl','r') as f:
    data = pkl.load(f)
print 'data.pkl'
print data
