# struct
cdef struct mystruct:
    int a
    double b

cdef mystruct var

var.a = 3
var.b = 0.7

print var
