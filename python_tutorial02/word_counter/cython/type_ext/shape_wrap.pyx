from libc.stdlib cimport malloc, free

cdef extern from "shape.h":
    cdef struct _Circle:
        double x,y
        double r

    ctypedef _Circle Circle

    cdef double area(Circle *c)
    cdef double circumference(Circle *c)
    cdef void center(Circle *c, double *x, double *y)

cdef class circle:
    cdef Circle *_pt
    def __cinit__(self,x,y,r):
        self._pt = <Circle*> malloc(sizeof(Circle))
        self._pt.x = x
        self._pt.y = y
        self._pt.r = r

    cpdef double area(self):
        return area(self._pt)

    cpdef double circumference(self):
        return circumference(self._pt)

    cpdef tuple center(self):
        cdef double x,y 
        center(self._pt, &x, &y)
        return (x,y)

    def __dealloc__(self):
        free(self._pt)
