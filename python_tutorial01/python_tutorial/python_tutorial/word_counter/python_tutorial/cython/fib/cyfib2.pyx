def fib(long n):
    cdef long a = 0
    cdef long b = 1
    cdef long i
    for i in range(n):
        a, b = a+b, a
    return a
