cdef extern from "shape.h":
    cdef struct _Circle:
        double x,y
        double r

    ctypedef _Circle Circle

    cdef double area(Circle *c)
    cdef double circumference(Circle *c)
    cdef void center(Circle *c, double *x, double *y)
