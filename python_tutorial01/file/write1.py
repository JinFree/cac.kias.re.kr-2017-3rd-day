#!/usr/bin/env python

def main():
    f = open('data.txt','w')
    f.write('This is the first line.\n')
    f.write('This is the second line.\n')
    f.close()

if __name__ == '__main__':
    main()

