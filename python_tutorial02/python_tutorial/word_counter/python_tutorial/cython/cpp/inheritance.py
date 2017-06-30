#!/usr/bin/env python
from shape_wrap import circle

class circle2(circle):
    def __init__(self,x,y,r):
        print "This is circle2!"
        super(circle2,self).__init__(x,y,r)

def main():
    a = circle2(1.,2.,3.)
    print a.area()

if __name__ == '__main__':
    main()

