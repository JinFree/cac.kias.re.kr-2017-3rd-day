import numpy as np

a = np.arange(30)
print a
print a.shape
a = a[np.newaxis,:]
print a
print a.shape
a = a.reshape((10,3))
print a 
print a.shape

b = a + 100

print 'vstack'
print np.vstack((a,b))
print 'hstack'
print np.hstack((a,b))
print 'concatenate'
print np.concatenate((a,b),axis=0)

print 'ravel'
print a.ravel() 

print 'transpose'
print a.T

c = a.copy()
print c
