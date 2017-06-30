# Not so simple class example

class A(object):
    def __init__(self):
        print "A"
        super(A, self).__init__()
        print "A'"

class K(object):
    def __init__(self):
        print "K"
        super(K, self).__init__()
        print "K'"

class B(A,K):
    def __init__(self):
        print "B"
        super(B, self).__init__()
        print "B'"

class C(object):
    def __init__(self):
        print "C"
        super(C, self).__init__()
        print "C'"

class D(B, C):
    def __init__(self):
        print "D"
        super(D, self).__init__()
        print "D'"

d = D()

print 'MRO'
print D.__mro__
