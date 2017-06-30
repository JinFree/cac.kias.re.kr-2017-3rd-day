#!/usr/bin/env python

def main():
    f = open('data.txt','r')
    for line in f:
        print line,        
    f.close()

if __name__ == '__main__':
    main()

