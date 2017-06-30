# union
cdef union myunion:
    int a
    double b
    char c

cdef myunion var
var.a = 1
print var
var.b = 1.0
print var
var.c = 'a'
print var
