# distutils: language=c++
from cython.operator cimport dereference as deref, preincrement as inc

cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        cppclass iterator:
            T operator*()
            iterator operator++()
            bint operator==(iterator)
            bint operator!=(iterator)
        vector()
        void push_back(T&)
        T& operator[](int)
        T& at(int)
        iterator begin()
        iterator end()

cdef vector[int] *v = new vector[int]()
cdef int i
for i in range(0,10,2):
    v.push_back(i)

cdef vector[int].iterator it = v.begin()
while it != v.end():
    print deref(it)
    inc(it) # it += 1

# print v # ERROR
del v
