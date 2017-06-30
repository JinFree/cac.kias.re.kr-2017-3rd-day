from libc.stdlib cimport malloc, free
from circle_st cimport *

cdef class circle:
    cdef Circle *_pt
    def __cinit__(self,x,y,r):
        self._pt = <Circle*> malloc(sizeof(Circle))
        self._pt.x = x
        self._pt.y = y
        self._pt.r = r

    def area(self):
        return area(self._pt)

    def circumference(self):
        return circumference(self._pt)

    def center(self):
        cdef double x,y 
        center(self._pt, &x, &y)
        return (x,y)

    def __dealloc__(self):
        free(self._pt)
