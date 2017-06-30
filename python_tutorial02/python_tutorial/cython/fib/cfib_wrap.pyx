cdef extern from "cfib.h" nogil:
    long cfib(long n) 

def fib(int n):
    cdef long ret 
    ret = cfib(n)
    return ret
