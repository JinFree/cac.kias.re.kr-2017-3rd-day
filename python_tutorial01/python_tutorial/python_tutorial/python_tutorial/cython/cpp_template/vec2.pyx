from libcpp.vector cimport vector

cdef vector[int] vect = range(1, 10, 2)
print vect

# del vect # ERROR!
