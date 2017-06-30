# distutils: language = c++

cdef extern from "shape.h" namespace "shape":
    cdef cppclass Circle:
        Circle(double xin, double yin, double rin)
        double area()
        double circumference()
        void center(double *xout, double *yout)

cdef class circle:
    cdef Circle *_pt

    def __cinit__(self,x,y,r):
        self._pt = new Circle(x,y,r)

    def area(self):
        return self._pt.area()

    def circumference(self):
        return self._pt.circumference()

    def center(self):
        cdef double x,y 
        self._pt.center(&x, &y)
        return (x,y)

    def __dealloc__(self):
        del self._pt
