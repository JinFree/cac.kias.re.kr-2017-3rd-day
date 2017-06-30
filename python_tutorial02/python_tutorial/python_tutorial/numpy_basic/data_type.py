#!/usr/bin/env python
import numpy as np

# integer type
print 'INTEGER DATA TYPE'
integer_data = np.arange(10)
print integer_data
print integer_data.dtype
print

# float (real number) type
print 'FLOAT DATA TYPE'
float_data = np.array( [ 10.4, 8.3, 1.9, 2.2 ] )
print float_data
print float_data.dtype
print

# complex number type
print 'COMPLEX DATA TYPE'
complex_data = np.array( [1.0+2.0j, 0.0+0.0j, 7.0+1.2j] )
print complex_data
print complex_data.real
print complex_data.imag
print complex_data.dtype
print complex_data.real.dtype
print complex_data.imag.dtype
print

# Bool type
print 'BOOL DATA TYPE'
bool_data = np.ones(5,dtype=np.bool)
print bool_data
print bool_data.dtype
bool_data2 = np.zeros(5,dtype=np.uint8)
print bool_data2
print bool_data2.dtype
